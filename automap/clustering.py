import numpy as np
import ase

from automap.utils import get_cluster_positions, get_cluster_elements


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

    def apply(self, pair):
        """Clusters a pair of particles

        Arguments
        ---------

        pair (tuple):
            tuple containing two integers that refer to the particles
            which should be clustered.

        """
        raise NotImplementedError

    @staticmethod
    def _cluster_pair(pair, clusters, projection, indices, masses):
        """Clusters a pair of particles in-place

        Arguments
        ---------

        pair (tuple):
            tuple containing two integers that refer to the particles
            which should be clustered.

        clusters (ndarray of shape (natom, natom)):
            description of current cluster configuration

        """
        pass

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
        pass

    def get_indices(self):
        """Returns the atom indices for each cluster (as a tuple)"""
        indices = []
        natom = 0
        for i in range(self.get_ncluster()):
            nonzero = self.clusters[i].nonzero()
            assert len(nonzero) > 0
            indices.append(nonzero)
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
            - if each nonzero row in self.clusters_mw is normalized.
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
        return True
