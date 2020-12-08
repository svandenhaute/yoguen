import ase
import numpy as np

from tqdm import tqdm

from automap.utils import compute_entropy_quantum, compute_entropy_classical, \
        get_mass_matrix, get_internal_basis, expand_mapping
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
        masses = self.atoms.get_masses()
        for i, group in enumerate(indices):
            self.clusters[i, np.array(group)] = 1
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

    def _score_pairs(self, quadratic, pairs, T=300, progress=False):
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

        T (double):
            temperature at which the entropy should be computed, in kelvin.

        progress (bool):
            specifies whether or not a progress bar will be shown that
            indicates the progress

        """
        # general stuff
        masses   = self.atoms.get_masses().copy()
        indices  = self.get_indices()
        ncluster = self.get_ncluster()
        smap     = np.zeros(len(pairs))

        # create transformation arrays that remain fixed throughout the list
        W_r = get_mass_matrix(self.atoms)
        B_r = get_internal_basis(self.atoms, mw=True)
        assert np.allclose(B_r.T @ B_r, np.identity(B_r.shape[1]))
        t_mw = np.ones((1, len(self.atoms))) @ np.sqrt(np.diag(masses))
        _, __, vH = np.linalg.svd(t_mw)
        B_r_small = np.transpose(vH[1:])
        B_r_ = expand_mapping(B_r_small)
        assert np.allclose(B_r, B_r_)

        # precompute transformed hessians
        hessian    = quadratic.hessian.copy()
        hessian_m  = np.linalg.inv(np.sqrt(W_r)) @ hessian @ np.linalg.inv(np.sqrt(W_r))
        hessian_ic = np.transpose(B_r) @ hessian_m @ B_r
        for k, pair in tqdm(enumerate(pairs), total=len(pairs), unit='pairs', disable=not progress):
            _indices = self._join_pair(indices, pair)
            _masses     = np.zeros(ncluster - 1)
            _projection = np.zeros((ncluster - 1, len(self.atoms)))
            for i, group in enumerate(_indices):
                _masses[i] = np.sum(masses[np.array(group)])
                for j in group:
                    _projection[i, j] = np.sqrt(masses[j]) / np.sqrt(_masses[i])

            # transform _projection and use svd to obtain generalized basis
            T_mw = np.ones((1, ncluster - 1)) @ np.sqrt(np.diag(_masses))
            _, __, vH = np.linalg.svd(T_mw)
            B_R_small = np.transpose(vH[1:])
            _, sigmas, basis_small = np.linalg.svd(
                    B_R_small.T @ _projection @ B_r_small, # transform
                    )
            assert np.allclose(sigmas, np.ones(sigmas.shape)) # check sigmas
            KN = np.transpose(expand_mapping(basis_small)) # triple size
            assert np.allclose(KN @ KN.T, np.identity(KN.shape[0])) # ortho
            assert np.allclose(KN.T @ KN, np.identity(KN.shape[0])) # ortho

            # transform hessian into generalized row space, create blocks
            hessian_row = np.transpose(KN) @ hessian_ic @ KN
            size = 3 * _projection.shape[0] - 3 # depends on periodicity
            #hessian_11 = hessian_row[:size, :size]
            #hessian_12 = hessian_row[:size, size:]
            hessian_22  = hessian_row[size:, size:]
            omegas, _   = np.linalg.eigh(hessian_22)
            frequencies = np.sqrt(omegas) / (2 * np.pi)
            smap[k]     = np.sum(compute_entropy_quantum(frequencies, T))
        return smap

    @staticmethod
    def _join_pair(indices, pair):
        """Returns new indices with joined clusters"""
        _indices = list(indices)
        _pair = list(pair)
        _pair.sort() # ensure smallest index is at _pair[0]
        _indices[_pair[0]] = _indices[_pair[0]] + _indices[_pair[1]]
        _indices.pop(_pair[1])
        return tuple(_indices)

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
        pos_c = self.get_cluster_positions()

        # elements
        numbers_c = self.get_cluster_elements()
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

    def get_cluster_positions(self):
        """Computes the positions of the clusters"""
        ncluster = self.get_ncluster()
        indices  = self.get_indices()
        pos_c    = np.zeros((ncluster, 3))
        pos      = self.atoms.get_positions()
        masses   = self.atoms.get_masses()

        for i, group in enumerate(indices):
            # compute total mass of group
            mass = np.sum(masses[np.array(group)])

            # first atom is used as reference. COM is computed using relative
            # vectors only, in order to apply the mic consistently 
            index_ref = group[0]
            pos_c[i, :] = pos[index_ref, :]
            for j in range(1, len(group)):
                index = group[j]
                delta = self.atoms.get_distance(index_ref, index, mic=True, vector=True)
                pos_c[i, :] += masses[index] / mass * delta
        return pos_c

    def get_cluster_elements(self):
        """Returns elements of clusters

        Clusters that are chemically equivalent should have the same element.
        Clusters containing only one atom should have the element of that atom.
        Cluster elements start at 118 and count backward.

        """
        ncluster  = self.get_ncluster()
        indices   = self.get_indices()
        numbers   = self.atoms.get_atomic_numbers()
        numbers_c = np.zeros(ncluster)

        cluster_elements = {}
        new_key = 118 # last element in periodic table

        for i, group in enumerate(indices):
            if len(group) == 1: # if only one atom, then number is same
                numbers_c[i] = numbers[group[0]]
            else: # if multiple atoms, then start at 118
                numbers_in_group = set(numbers[np.array(group)])
                found = False
                for key, value in cluster_elements.items(): # iterate over prev
                    if value == numbers_in_group:
                        numbers_c[i] = key
                        found = True
                if not found:
                    cluster_elements[new_key] = numbers_in_group
                    numbers_c[i] = new_key
                    new_key -= 1
        return numbers_c

    def get_elements_in_cluster(self, index):
        """Returns the atomic elements within a given cluster"""
        indices = self.get_indices()
        group = indices[index]
        symbols = list(self.atoms.symbols)
        cluster_symbols = set([symbols[i] for i in group])
        return cluster_symbols

    def validate(self):
        """Validates the current clustering

        This function checks:
            - if each nonzero row in self.clusters is mutually orthogonal with
              every other row
            - if each column contains exactly one nonzero entry.

        """
        # entries should be either zero or one
        assert np.all((self.clusters == 1) + (self.clusters == 0))

        # check whether each particle belongs to precisely one cluster
        # (automatically satisfies orthogonality constraint)
        assert np.allclose(np.ones(len(self.atoms)), np.sum(self.clusters, axis=0))

        # checks indices calculation
        indices = self.get_indices()
        for i, group in enumerate(indices):
            assert np.all(self.clusters[i, np.array(group)])

        # create XYZ representation of clustering and verify
        clusters_XYZ = expand_mapping(self.clusters)
        for i, group in enumerate(indices):
            for j in range(len(self.atoms)):
                assert self.clusters[i, j] == clusters_XYZ[3 * i    , 3 * j    ]
                assert self.clusters[i, j] == clusters_XYZ[3 * i + 1, 3 * j + 1]
                assert self.clusters[i, j] == clusters_XYZ[3 * i + 2, 3 * j + 2]
        return True

    def visualize(self, path_file):
        """Visualizes the current clustering configuration

        It writes an .xyz file containing all initial atoms, but with a
        coloring that indicates the different clusters.

        """
        _atoms = self.atoms.copy()
        numbers_clusters = self.get_cluster_elements()
        numbers_visual   = np.zeros(len(_atoms))
        for i in range(len(_atoms)):
            index_cluster = np.where(self.clusters[:, i])[0]
            numbers_visual[i] = numbers_clusters[index_cluster]
        _atoms.set_atomic_numbers(numbers_visual)
        _atoms.write(path_file)
        return _atoms

    def complete_clusters_by_translation(self):
        """This function translates atoms to complete clusters"""
        indices   = self.get_indices()
        positions = self.atoms.get_positions()
        for i, group in enumerate(indices):
            index_ref = group[0] # reference atom for the deltas
            pos_ref   = positions[index_ref]
            for j in group[1:]:
                delta = self.atoms.get_distance(
                        index_ref,
                        j,
                        mic=True,
                        vector=True,
                        )
                positions[j, :] = pos_ref + delta
        self.atoms.set_positions(positions)
