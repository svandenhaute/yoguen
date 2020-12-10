import logging
import numpy as np

from abc import ABC, abstractmethod

from yoguen.utils import get_natom_from_indices


logger = logging.getLogger(__name__) # logging per module


class Candidate(ABC):
    """Base class to represent a combination of clusters"""

    @abstractmethod
    def apply(self, indices):
        """Applies the candidate and returns a copy of indices"""
        pass

    @abstractmethod
    def log(self):
        """logs basics properties of candidate"""

    @staticmethod
    @abstractmethod
    def apply_clist(clist, indices):
        """Applies list of Candidate instances to indices"""
        pass

    @staticmethod
    @abstractmethod
    def possibly_equivalent(c0, c1, **kwargs):
        """Assesses whether two candidates are possibly equivalent"""
        pass


class Pair(Candidate):
    """Represents a pair of clusters"""

    def __init__(self, i, j, identities, distance):
        """Constructor

        Arguments
        ---------

        i (int):
            index of first cluster in pair.

        j (int):
            index of second cluster in pair.

        identities (tuple):
            tuple of length two describing the identities of each cluster.
            Each identity is again a tuple (see yoguen.clustering)

        distance (double):
            distance between the clusters, in angstrom.

        """
        assert type(i) == int
        assert type(j) == int
        assert i != j
        assert len(identities) == 2
        if i < j: # ensure self._i is smallest index
            self._i = i
            self._j = j
            self.identities = identities
        else:
            self._i = j
            self._j = i
            self.identities = identities[::-1]
        self.distance = distance

    def apply(self, indices):
        """join two clusters"""
        indices_ = list(indices) # make a copy
        indices_[self._i] = indices_[self._i] + indices_[self._j]
        indices_.pop(self._j) # self._i < self._j
        return indices_

    def log(self):
        logger.info('pair {}'.format((self[0], self[1])))
        #logger.info('\ttypes {} and {}'.format(symbols[0], symbols[1]))
        logger.info('\tdistance between clusters is {:.5e} angstrom'.format(self.distance))
        #logger.info('\tjoining atomic indices {} and {}'.format(groups[0], groups[1]))
        logger.info('\twith cluster identities {} and {}'.format(
            self.identities[0], self.identities[1]))

    @staticmethod
    def apply_clist(clist, indices):
        natom             = get_natom_from_indices(indices)
        modified_clusters = np.array([False] * len(indices)) # changed clusters
        remove_clusters   = np.array([False] * len(indices)) # to remove
        new_groups        = {} # keep track of newly formed groups
        indices_          = []
        for i, pair in enumerate(clist):
            new_groups[pair[0]] = indices[pair[0]] + indices[pair[1]]
            modified_clusters[pair[0]] = True
            remove_clusters[pair[1]]   = True
        for i, group in enumerate(indices):
            if modified_clusters[i]: # modified group; add from new_groups
                assert i in new_groups.keys()
                indices_.append(tuple(new_groups[i]))
            elif remove_clusters[i]: # obsolete group; do not add
                pass
            else: # unchanged group; add
                indices_.append(tuple(group))
        return indices_

    @staticmethod
    def possibly_equivalent(c0, c1, tol_distance=1e-3):
        delta = 2 * (c0.distance - c1.distance) / (c0.distance + c1.distance)
        equiv = np.all([
            c0.identities[0] == c1.identities[0], # cluster element
            tuple(c0.identities[1]) == tuple(c1.identities[1]), # elements in cluster
            abs(delta) < tol_distance, # symmetric relative difference in dist
            ])
        return equiv

    @classmethod
    def get_pair(cls, clustering, i, j):
        """Create a pair from a clustering

        Arguments
        ---------

        clustering (``Clustering`` instance):
            used to determine the identity of this pair,

        i (int):
            index of first cluster in pair

        j (int):
            index of second cluster in pair

        """
        if clustering.identities is None:
            clustering.get_identities() # generate if not yet present
        # access list directly to avoid creating copy each time
        identities = tuple([
            clustering.identities[i],
            clustering.identities[j],
            ])
        atoms_reduced = clustering.get_atoms_reduced()
        distance = atoms_reduced.get_distance(
                i,
                j,
                mic=True,
                )
        return cls(int(i), int(j), identities, distance)

    def __getitem__(self, key):
        assert type(key) == int
        if (key != 1) and (key != 0):
            raise IndexError
        if key == 0:
            return self._i
        else:
            return self._j
