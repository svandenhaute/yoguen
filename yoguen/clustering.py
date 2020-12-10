import ase
import numpy as np

from tqdm import tqdm

from yoguen.utils import compute_entropy_quantum, compute_entropy_classical, \
        get_mass_matrix, get_internal_basis, expand_mapping
from yoguen.models import Quadratic


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
        self.indices     = [tuple([i]) for i in range(len(self.atoms))]
        self.clusters    = np.eye(len(self.atoms), dtype=np.dtype(int))

        # create attribute to store element information for each cluster
        # and the atoms_reduced object used for IO and distance calculations
        self.identities    = None
        self.atoms_reduced = None

    def update_indices(self, indices):
        """Rebuilds the clusters array based on new indices

        Arguments
        ---------

        indices (tuple of tuples):
            tuple of atom index tuples that describes the clustering

        """
        # invalidate attributes
        self.identities    = None
        self.atoms_reduced = None

        self.indices = list(indices) # create copy
        self.clusters[:]   = 0
        masses = self.atoms.get_masses()
        for i, group in enumerate(indices):
            self.clusters[i, np.array(group)] = 1
        self.validate() # validate current clustering

    def apply(self, quadratic, temperature=300):
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

        temperature (double):
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
        smap_quantum   = np.sum(compute_entropy_quantum(frequencies, temperature))
        smap_classical = np.sum(compute_entropy_classical(frequencies, temperature))
        saa_quantum    = quadratic.compute_entropy(temperature, quantum=True)
        saa_classical  = quadratic.compute_entropy(temperature, quantum=False)

        # compute mass-weighted CG hessian
        # compute classical and quantum entropies, verify results
        hessian_      = hessian_11
        hessian_     -= hessian_12 @ np.linalg.inv(hessian_22) @ hessian_12.T
        omegas, _     = np.linalg.eigh(hessian_)
        frequencies   = np.sqrt(omegas) / (2 * np.pi)
        scg_quantum   = np.sum(compute_entropy_quantum(frequencies, temperature))
        scg_classical = np.sum(compute_entropy_classical(frequencies, temperature))
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

    def score_candidates(self, clist, quadratic, temperature=300,
            progress=False):
        """Computes the mapping entropy for each of the cluster pair

        This function considers pairs of clusters and computes the mapping
        entropy that would be encountered when these clusters would be merged.
        It contains a few optimizations in comparison to self.apply() which
        sacrifice modularity and readability for speed.

        Arguments
        ---------

        clist (list of ``Candidate`` subclass instances):
            list of candidates

        quadratic (``Quadratic`` instance):
            quadratic which describes the PES in the original degrees of
            freedom. Its ``Atoms`` instance should be the same as self.atoms.

        temperature (double):
            temperature at which the entropy should be computed, in kelvin.

        progress (bool):
            specifies whether or not a progress bar will be shown that
            indicates the progress

        """
        masses   = self.atoms.get_masses().copy()
        ncluster = self.get_ncluster()
        smap     = np.zeros(len(clist))

        # precompute transformed hessian
        mass_weighting = np.linalg.inv(np.sqrt(get_mass_matrix(self.atoms)))
        hessian        = quadratic.hessian.copy()
        hessian_m      = mass_weighting @ hessian @ mass_weighting

        # add progress bar
        iterator = tqdm(
                enumerate(clist),
                total=len(clist),
                unit='candidates',
                disable=not progress,
                )
        for k, candidate in iterator:
            _indices = candidate.apply(self.indices) # obtain hypothet. indices
            _masses     = np.zeros(ncluster - 1)
            _projection = np.zeros((ncluster - 1, len(self.atoms)))
            for i, group in enumerate(_indices):
                _masses[i] = np.sum(masses[np.array(group)])
                for j in group:
                    _projection[i, j] = np.sqrt(masses[j]) / np.sqrt(_masses[i])

            # use svd
            _, sigmas, basis_small = np.linalg.svd(_projection)
            N_small      = np.transpose(basis_small)[:, ncluster - 1:]
            N            = expand_mapping(N_small)
            hessian_null = np.transpose(N) @ hessian_m @ N
            omegas, _    = np.linalg.eigh(hessian_null)
            frequencies  = np.sqrt(omegas) / (2 * np.pi)
            smap[k]      = np.sum(
                    compute_entropy_quantum(frequencies, temperature),
                    )
            assert np.allclose(sigmas, np.ones(sigmas.shape)) # check sigmas
        return smap

    def get_atoms_reduced(self):
        """Constructs an ``Atoms`` instance for the clustered system"""
        if self.atoms_reduced is None:
            masses          = self.atoms.get_masses()
            masses_clusters  = np.zeros(len(self.indices)) # cluster masses
            numbers_clusters = np.zeros(len(self.indices))
            pos_clusters     = self.get_cluster_positions()

            # fill arrays
            for i, group in enumerate(self.indices):
                masses_clusters[i] = np.sum(masses[np.array(group)])
            assert np.all(masses_clusters > 0) # masses are strictly positive
            if self.identities is None:
                self.get_identities()
            for i in range(len(self.indices)):
                numbers_clusters[i] = self.identities[i][0]

            self.atoms_reduced = ase.Atoms(
                    numbers=numbers_clusters,
                    positions=pos_clusters,
                    cell=self.atoms.get_cell(),
                    masses=masses_clusters,
                    pbc=True, # apply PBCs along all three dimensions
                    )
        return self.atoms_reduced

    def get_mapping(self):
        """Constructs the mapping matrix"""
        masses  = self.atoms.get_masses()
        mapping = np.zeros((3 * self.get_ncluster(), 3 * len(self.atoms)))
        for i, group in enumerate(self.indices):
            weights = masses[np.array(group)]
            weights /= np.sum(weights)
            for j, atom in enumerate(group):
                mapping[3 * i, 3 * atom] = weights[j]
                mapping[3 * i + 1, 3 * atom + 1] = weights[j]
                mapping[3 * i + 2, 3 * atom + 2] = weights[j]
        return mapping

    def get_ncluster(self):
        """Returns the number of clusters

        This is computed as the number of nonzero rows in the self.clusters
        array.

        """
        return np.sum(np.any(self.clusters, axis=1))

    def get_cluster_positions(self):
        """Computes the positions of the clusters"""
        ncluster = self.get_ncluster()
        pos_c    = np.zeros((ncluster, 3))
        pos      = self.atoms.get_positions()
        masses   = self.atoms.get_masses()

        for i, group in enumerate(self.indices):
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

    def get_identities(self):
        """Returns a list of cluster identities

        Each identity is a tuple with two items. The first item is an integer
        representing the cluster chemical element. This is a fictional
        association that is only used when saving the reduced representation
        to a file format such as .xyz. The second item is a set of atomic
        numbers that are present in the cluster.

        The cluster chemical element is determined as follows
            -   if the cluster contains only a single atom, then its chemical
                element is simply taken from the atom
            -   if the cluster contains more than one atom, then its chemical
                element is assigned randomly. To avoid interference with other
                atoms in the system, clusters are assigned elements starting
                at the largest atomic element (number 118) and counting down.
                Two clusters whose cluster_elements sets are identical, are
                assigned to the same element.

        """
        if self.identities is None: # build cluster identities
            numbers = self.atoms.get_atomic_numbers()
            random_cluster_number = 118 # assign cluster numbers start
            self.identities = []
            for i, group in enumerate(self.indices):
                cluster_element = None
                atomic_elements = None
                if len(group) == 1: # if only one atom, then number is same
                    atomic_elements = set([numbers[group[0]]])
                    cluster_element = numbers[group[0]]
                else:
                    atomic_elements = set(numbers[np.array(group)])
                    for (_element, _elements) in self.identities:
                        if atomic_elements == _elements:
                            cluster_element = _element
                    if cluster_element is None: # cluster not found
                        cluster_element = random_cluster_number
                        random_cluster_number -= 1
                        # avoid overlap;
                        assert random_cluster_number > numbers.max()
                assert cluster_element is not None
                assert atomic_elements is not None
                identity = tuple([cluster_element, set(atomic_elements)])
                self.identities.append(identity)
        return list(self.identities) # return copy

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

        # checks consistency between indices and clusters
        for i, group in enumerate(self.indices):
            assert np.all(self.clusters[i, np.array(group)])

        # create XYZ representation of clustering and verify
        clusters_XYZ = expand_mapping(self.clusters)
        for i, group in enumerate(self.indices):
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

        atoms_reduced    = self.get_atoms_reduced()
        numbers_clusters = atoms_reduced.get_atomic_numbers()
        numbers_visual   = np.zeros(len(_atoms))
        for i in range(len(_atoms)):
            index_cluster = np.where(self.clusters[:, i])[0]
            numbers_visual[i] = numbers_clusters[index_cluster]
        _atoms.set_atomic_numbers(numbers_visual)
        _atoms.write(path_file)
        return _atoms

    def complete_clusters_by_translation(self):
        """This function translates atoms to complete clusters"""
        positions = self.atoms.get_positions()
        for i, group in enumerate(self.indices):
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
