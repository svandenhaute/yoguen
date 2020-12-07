import numpy as np
import ase

from automap.utils import get_cluster_positions, get_cluster_elements, \
        compute_entropy_quantum, compute_entropy_classical, get_mass_matrix, \
        get_internal_basis, expand_mapping
from automap.models import Quadratic


class Clustering(object):
    """Represents a clustering of atoms

    Attributes
    ----------

    atoms (``Atoms`` instance):
        atoms instance that determines the initial number of particles and
        their masses, together with the geometry and cell configuration based
        on which the neighborlist objects will be constructed.

    """

    def __init__(self, atoms):
        """Constructor

        Arguments
        ---------

        atoms (``Atoms`` instance):
            Atoms instance of the system.

        """
        self.atoms = atoms

        # clustering information is stored in ndof x ndof array for efficiency
        # for each clustering, only the first k rows will be accessed by using
        # the numpy view functionality.
        self.clusters    = np.eye(len(self.atoms), dtype=np.dtype(int))
        self.projection  = np.eye(len(self.atoms))

    def update_indices(self, indices):
        """Rebuilds the clusters and projection arrays based on new indices

        Arguments
        ---------

        indices (tuple of tuples):
            tuple of atom index tuples that describes the clustering

        """
        self.clusters[:]   = 0
        self.projection[:] = 0.0
        masses = self.atoms.get_masses()
        for i, group in enumerate(indices):
            self.clusters[i, np.array(group)] = 1
            total_mass = np.sum(masses[np.array(group)])
            for atom in group:
                self.projection[i, atom] = np.sqrt(masses[atom] / total_mass)
        self.validate() # validate current clustering

    def apply(self, quadratic, T=300):
        """Applies the clustering to generate the CG ``Quadratic`` instance

        Both the entropies involved (atomistic, mapping and CG entropies) as
        well as the actual CG quadratic are returned. The CG hessian is
        computed in the following way:
            (0) start with the mass-weighted hessian and mapping transformation
                arrays
            (1) compute eigenmodes and eigenvalues of hessian
            (2) generate internal basis that removes global translations (and
                rotations in case of nonperiodic systems) and transform hessian
                and mapping transformation.
            (3) compute the SVD of the mapping
            (4) transform the internal hessian in the generalized row space of
                the mapping; is then obtained based on the block submatrices

        The ``Quadratic`` constructor requires four arguments:
            -   an ``Atoms`` instance that represents the CG system. This is
                obtained using self.get_atoms_reduced().
            -   the CG hessian as calculated using matrix algebra (vide supra)
            -   the equilibrium geometry (present in atoms_reduced)
            -   the equilibrium cell matrix (present in atoms_reduced)

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            quadratic which describes the PES in the original degrees of
            freedom. Its ``Atoms`` instance should be the same as self.atoms.

        T (double):
            temperature at which the clustering should be applied, in kelvin.

        """
        assert id(self.atoms) == id(quadratic.atoms)
        # set equilibrium geometry and cell before computing atoms_reduced
        self.atoms.set_positions(quadratic.geometry)
        self.atoms.set_cell(quadratic.cell)
        atoms_reduced = self.get_atoms_reduced()

        # mass diagonal matrix for atomistic system
        W_r = get_mass_matrix(self.atoms)
        # internal basis for atomistic representation
        B_r = get_internal_basis(self.atoms, mw=True)

        # mass diagonal matrix for reduced system
        W_R = get_mass_matrix(atoms_reduced)
        # internal basis for reduced representation
        B_R = get_internal_basis(atoms_reduced, mw=True)

        # get arrays and apply mass-weighting
        mapping   = self.get_mapping()
        mapping_m = np.sqrt(W_R) @ mapping @ np.linalg.inv(np.sqrt(W_r))
        hessian   = quadratic.hessian.copy()
        hessian_m = np.linalg.inv(np.sqrt(W_r)) @ hessian @ np.linalg.inv(np.sqrt(W_r))

        # transform to internal coordinates
        mapping_ic = np.transpose(B_R) @ mapping_m @ B_r
        hessian_ic = np.transpose(B_r) @ hessian_m @ B_r
        _, sigmas, KNT = np.linalg.svd(mapping_ic)
        assert np.allclose(sigmas, np.ones(sigmas.shape)) # sing. values == 1
        KN = np.transpose(KNT) # generalized row space of mapping

        # transform hessian into generalized row space, create blocks
        hessian_row = np.transpose(KN) @ hessian_ic @ KN
        size = mapping_ic.shape[0] # depends on periodicity
        hessian_11 = hessian_row[:size, :size]
        hessian_12 = hessian_row[:size, size:]
        hessian_22 = hessian_row[size:, size:]

        # diagonalize lower right block to obtain frequencies
        # compute classical and quantum entropy
        omegas, _ = np.linalg.eigh(hessian_22)
        frequencies = np.sqrt(omegas) / (2 * np.pi)
        smap_quantum   = np.sum(compute_entropy_quantum(frequencies, T))
        smap_classical = np.sum(compute_entropy_classical(frequencies, T))
        saa_quantum    = quadratic.compute_entropy(T, quantum=True)
        saa_classical  = quadratic.compute_entropy(T, quantum=False)

        # compute mass-weighted CG hessian
        # compute classical and quantum entropies, verify results
        hessian_      = hessian_11
        hessian_     -= hessian_12 @ np.linalg.inv(hessian_22) @ hessian_12.T
        omegas, _     = np.linalg.eigh(hessian_)
        frequencies   = np.sqrt(omegas) / (2 * np.pi)
        scg_quantum   = np.sum(compute_entropy_quantum(frequencies, T))
        scg_classical = np.sum(compute_entropy_classical(frequencies, T))
        assert abs(saa_classical - (scg_classical + smap_classical)) < 1e-9

        # reverse transformations to obtain hessian for quadratic
        hessian_cg = B_R @ hessian_ @ B_R.T
        hessian_cg = np.sqrt(W_R) @ hessian_cg @ np.sqrt(W_R)
        return (saa_quantum, smap_quantum, scg_quantum), Quadratic(
                atoms_reduced,
                hessian_cg,
                atoms_reduced.get_positions(),
                atoms_reduced.get_cell(),
                )

    def _score_pairs(self, quadratic, pairs, T=300):
        """Computes the mapping entropy for each of the cluster pair

        This function considers pairs of clusters and computes the mapping
        entropy that would be encountered when these clusters would be merged.
        It contains a few optimizations in comparison to self.apply() which
        sacrifice modularity and readability for speed.

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            quadratic which describes the PES in the original degrees of
            freedom. Its ``Atoms`` instance should be the same as self.atoms.

        pairs (list of tuples):
            list of pairs of cluster indices.

        """
        # general stuff
        masses  = self.atoms.get_masses().copy()
        indices = self.get_indices()

        # create transformation arrays that remain fixed throughout the list
        W_r = get_mass_matrix(self.atoms)
        B_r = get_internal_basis(self.atoms, mw=True)

        # allocate arrays that are only slightly modified for each pair
        ncluster    = self.get_ncluster()
        _projection = np.zeros((ncluster - 1, len(self.atoms)))
        _mapping    = np.zeros((3 * (ncluster - 1), 3 * len(self.atoms)), dtype=np.dtype(int))
        _masses     = np.zeros(ncluster - 1)

        # precompute transformed hessians
        hessian    = quadratic.hessian.copy()
        hessian_m  = np.linalg.inv(np.sqrt(W_r)) @ hessian @ np.linalg.inv(np.sqrt(W_r))
        hessian_ic = np.transpose(B_r) @ hessian_m @ B_r

        for pair in pairs:
            # modify _clusters and _masses, fill _W_R
            print(pair)
            print('0')
            _indices = list(indices)
            _indices[pair[0]] = _indices[pair[0]] + _indices[pair[1]]
            _indices.pop(pair[1])
            for i, group in enumerate(_indices):
                _masses[i] = np.sum(masses[np.array(group)])
                for j in group:
                    _projection[i, j] = np.sqrt(masses[j]) / np.sqrt(_masses[i])
            print('1')
            _mapping = expand_mapping(_projection)
            print('2')

            # generate B_R and apply
            Tr = np.zeros((3, (ncluster - 1) * 3))
            Tr[0, 0::3] = 1
            Tr[1, 1::3] = 1
            Tr[2, 2::3] = 1
            Tr_mw = Tr @ np.sqrt(np.diag(np.repeat(_masses, 3)))
            _, _, vH = np.linalg.svd(Tr_mw)
            B_R = np.transpose(vH[3:])
            print('3')
            mapping_ic = np.transpose(B_R) @ _mapping @ B_r
            print('4')

            _, sigmas, KNT = np.linalg.svd(mapping_ic)
            assert np.allclose(sigmas, np.ones(sigmas.shape)) # sing. values == 1
            KN = np.transpose(KNT) # generalized row space of mapping
            print('5')

            # transform hessian into generalized row space, create blocks
            hessian_row = np.transpose(KN) @ hessian_ic @ KN
            size = mapping_ic.shape[0] # depends on periodicity
            hessian_11 = hessian_row[:size, :size]
            hessian_12 = hessian_row[:size, size:]
            hessian_22 = hessian_row[size:, size:]
            print('6')

            omegas, _ = np.linalg.eigh(hessian_22)
            frequencies = np.sqrt(omegas) / (2 * np.pi)
            smap_quantum   = np.sum(compute_entropy_quantum(frequencies, T))
            _projection[:] = 0.0
            _masses[:]     = 0.0
            _mapping[:]    = 0.0


    def get_atoms_reduced(self):
        """Constructs an ``Atoms`` instance for the clustered system"""
        # masses
        masses = self.atoms.get_masses()
        indices = self.get_indices()
        masses_c = np.zeros(self.get_ncluster()) # cluster masses
        for i, group in enumerate(indices):
            masses_c[i] = np.sum(masses[np.array(group)])
        assert np.all(masses_c > 0) # masses are strictly positive

        # positions
        pos_c = get_cluster_positions(self.atoms, self)

        # elements
        numbers_c = get_cluster_elements(self.atoms, self)
        return ase.Atoms(
                numbers=numbers_c,
                positions=pos_c,
                cell=self.atoms.get_cell(),
                masses=masses_c,
                pbc=True, # apply PBCs along all three dimensions
                )

    def get_mapping(self):
        """Constructs the mapping matrix"""
        masses  = self.atoms.get_masses()
        indices = self.get_indices()
        mapping = np.zeros((3 * self.get_ncluster(), 3 * len(self.atoms)))
        for i, group in enumerate(indices):
            weights = masses[np.array(group)]
            weights /= np.sum(weights)
            for j, atom in enumerate(group):
                mapping[3 * i, 3 * atom] = weights[j]
                mapping[3 * i + 1, 3 * atom + 1] = weights[j]
                mapping[3 * i + 2, 3 * atom + 2] = weights[j]
        return mapping

    def get_indices(self):
        """Returns the atom indices for each cluster (as a tuple)"""
        indices = []
        natom = 0
        for i in range(self.get_ncluster()):
            nonzero = self.clusters[i].nonzero()[0]
            assert len(nonzero) > 0
            indices.append(tuple(nonzero))
            natom += len(nonzero)
        assert natom == len(self.atoms)
        return tuple(indices)

    def get_ncluster(self):
        """Returns the number of clusters

        This is computed as the number of nonzero rows in the self.clusters
        array.

        """
        return np.sum(np.any(self.clusters, axis=1))

    def validate(self):
        """Validates the current clustering

        This function checks:
            - if each nonzero row in self.clusters is mutually orthogonal with
              every other row
            - if each nonzero row in self.projection is normalized.
            - if each column contains exactly one nonzero entry.
            - if self.clusters and self.projection are equivalent

        """
        # check signs of all entries are positive and smaller than one
        assert np.all((self.clusters == 1) + (self.clusters == 0))
        assert np.all(self.projection >= 0)
        assert np.all(self.projection <= 1)

        # check whether each particle belongs to precisely one cluster
        # (automatically satisfies orthogonality constraint)
        assert np.allclose(np.ones(len(self.atoms)), np.sum(self.clusters, axis=0))

        # check whether coefficients are correct, by computing mass-weighted
        # clustering and verifying that rows are normalized
        ncluster = self.get_ncluster()
        assert np.allclose(
                np.linalg.norm(self.projection, axis=1)[:ncluster],
                np.ones(ncluster),
                )

        # check whether clusters and projection are consistent
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if self.clusters[i, j] != 0:
                    assert self.projection[i, j] > 0

        # checks indices calculation
        indices = self.get_indices()
        for i, group in enumerate(indices):
            assert np.all(self.clusters[i, np.array(group)])

        # create expanded mapping and verify
        mapping = expand_mapping(self.projection)
        for i, group in enumerate(indices):
            for j in range(len(self.atoms)):
                assert self.projection[i, j] == mapping[3 * i    , 3 * j    ]
                assert self.projection[i, j] == mapping[3 * i + 1, 3 * j + 1]
                assert self.projection[i, j] == mapping[3 * i + 2, 3 * j + 2]
        return True
