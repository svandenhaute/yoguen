import numpy as np


class Clustering(object):
    """Represents a clustering of atoms"""

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
        self.clusters     = np.eye(len(self.atoms))
        self.clusters_mw  = np.eye(len(self.atoms))
        # allocate memory for working copies
        self._clusters    = np.eye(len(self.atoms))
        self._clusters_mw = np.eye(len(self.atoms))

    def reduce(self, quadratic, threshold):
        """Cluster atoms until number of particles reaches threshold

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            represents the local curvature based on which an optimal clustering
            should be computed.

        """
        pass

    def compute_mapping_entropy(self, quadratic):
        """Computes the mapping entropy

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            represents the local curvature for which the current clustering
            should be evaluated.

        """
        basis = quadratic.get_conversion(kind='reduced')

    def get_indices(self):
        """Returns the atom indices for each cluster (as a tuple)"""
        indices = []
        natom = 0
        for i in range(self.get_ncluster()):
            nonzero = self._clusters[i].nonzero()
            assert len(nonzero) > 0
            indices.append(nonzero)
            natom += len(nonzero)
        assert 3 * natom == self.ndof
        return tuple(indices)

    def get_ncluster(self):
        """Returns the number of clusters

        This is computed as the number of nonzero rows in the self._clusters
        array.

        """
        return np.sum(np.any(self._clusters, axis=1))

    def validate(self):
        """Validates the current clustering

        This function checks:
            - if each nonzero row in self._clusters is mutually orthogonal with
              every other row
            - if each nonzero row is normalized.
            - if each column contains exactly one nonzero entry.

        """
        # check whether each particle belongs to precisely one cluster
        # (automatically satisfies orthogonality constraint)
        nonzero_entries_per_column = np.sum(np.abs(self._clusters) > 0, axis=0)
        assert np.allclose(np.ones(len(self.atoms)), nonzero_entries_per_column)

        # check whether coefficients are correct, by computing mass-weighted
        # clustering and verifying that rows are normalized
        ncluster = self.get_ncluster()
        assert np.allclose(
                np.linalg.norm(self._clusters_mw, axis=1)[:ncluster],
                np.ones(ncluster),
                )
        return True
