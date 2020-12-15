import logging
import numpy as np

from abc import ABC, abstractmethod
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

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

    @classmethod
    @abstractmethod
    def generate(cls, clustering):
        """Constructs a ``Candidate`` instances based on a clustering"""
        pass


class Pair:
    """Represents a pair of clusters"""

    def __init__(self, i, j, types, atom_indices, atom_types, distance,
            score=None):
        """Constructor

        Arguments
        ---------

        i (int):
            index of first cluster in pair.

        j (int):
            index of second cluster in pair.

        types (list of str):
            contains the 'atom' type assigned to each cluster

        atom_indices (list of tuples):
            contains the atom types present in each cluster

        distance (scalar):
            distance between the clusters, in angstrom

        score (scalar):
            positive scalar assigned to this pair

        """
        assert type(i) == int
        assert type(j) == int
        assert i != j
        assert len(types) == 2
        assert len(atom_indices) == 2
        assert len(atom_types) == 2
        self.types        = types
        self.atom_indices = atom_indices
        self.atom_types   = atom_types
        self.distance     = distance
        self.score        = score
        if score is not None:
            assert score > 0 # score should be strictly positive
        if i < j: # ensure self._i is smallest index
            self._i = i
            self._j = j
        else:
            self._i = j
            self._j = i
            self.types = types[::-1]
            self.atom_indices.reverse()
            self.atom_types.reverse()

    def apply(self, indices):
        indices_ = list(indices)
        indices_[self._i] = indices_[self._i] + indices_[self._j]
        indices_.pop(self._j) # self._i < self._j
        return indices_

    def log(self):
        logger.info('cluster indices {}, {}'.format(self._i, self._j))
        #logger.info('\ttypes {} and {}'.format(symbols[0], symbols[1]))
        logger.info('\tdistance between clusters is {:.5e} angstrom'.format(self.distance))
        #logger.info('\tjoining atomic indices {} and {}'.format(groups[0], groups[1]))
        logger.info('\tcluster types are {} and {}'.format(*self.types))
        logger.info('\twith atom types {} and {}'.format(
            self.atom_types[0], self.atom_types[1]))
        logger.info('\twith atom indices {} and {}'.format(
            self.atom_indices[0], self.atom_indices[1]))
        if self.score is not None:
            logger.info('\tand score: {:.5e}'.format(self.score))

    @staticmethod
    def test_equivalence(pair0, pair1, tol_distance=1e-3):
        """Returns whether pairs with different indices are equivalent

        For this to be the case, their types and distance should be identical.
        Atom indices are naturally different; atom types are identical if types
        are identical.

        """
        delta = (2 * (pair0.distance - pair1.distance)
                / (pair0.distance + pair1.distance))
        equal_types = ((tuple(pair0.types) == tuple(pair1.types)) or
                      (tuple(pair0.types) == tuple(pair1.types[::-1])))
        equal_distance = (abs(delta) <= tol_distance)
        return (equal_types and equal_distance)

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
        atom_indices = [
                clustering.indices[i],
                clustering.indices[j],
                ]
        if clustering.identities is None:
            clustering.get_identities() # generate if not yet present
        types = [
                clustering.identities[i][0],
                clustering.identities[j][0],
                ]
        atom_types = [
                clustering.identities[i][1],
                clustering.identities[j][1],
                ]
        atoms_reduced = clustering.get_atoms_reduced()
        distance = atoms_reduced.get_distance(
                i,
                j,
                mic=True,
                )
        return cls(int(i), int(j), types, atom_indices, atom_types, distance)

    def __getitem__(self, key):
        assert type(key) == int
        if (key != 1) and (key != 0):
            raise IndexError
        if key == 0:
            return self._i
        else:
            return self._j


class PairList(Candidate):
    """Represents a list of cluster pairs"""

    def __init__(self, pairs):
        """Constructor

        Arguments
        ---------

        pairs (list of ``Pair`` instances):
            list of pairs

        """
        self._pairs = list(pairs)
        self.npairs = len(pairs)

    @classmethod
    def generate(cls, clustering, cutoff, max_neighbors):
        """Generates a ``PairList`` instance based on a given clustering"""
        nlist = NeighborList(
                cutoff * np.ones(clustering.get_ncluster()) / 2,
                sorted=False,
                self_interaction=False,
                bothways=True,
                skin=0.0,
                primitive=NewPrimitiveNeighborList,
                )
        atoms_reduced = clustering.get_atoms_reduced()
        nlist.update(atoms_reduced) # construct neighborlist
        pair_tuples = []
        for i in range(len(atoms_reduced)):
            neighbors, _ = nlist.get_neighbors(i) # contains duplicates
            distances = np.array([atoms_reduced.get_distance(i, a, mic=True) for a in list(neighbors)])
            sorting = distances.argsort()
            # start from nearest neighbors and add up to self.max_neighbors,
            # but avoid duplicates
            npairs_to_add = min(len(neighbors), max_neighbors)
            npairs_added  = 0
            for j in range(len(sorting)):
                pair = [i, neighbors[sorting[j]]]
                pair.sort()
                if npairs_added < npairs_to_add:
                    if tuple(pair) not in pair_tuples:
                        pair_tuples.append(tuple(pair))
                        npairs_added += 1

        # generate list of Pair objects
        pairs = []
        for pair_tuple in pair_tuples:
            pairs.append(Pair.get_pair(clustering, *pair_tuple))
        return cls(pairs)

    def apply(self, indices):
        modified_clusters = np.array([False] * len(indices)) # changed clusters
        remove_clusters   = np.array([False] * len(indices)) # to remove
        new_groups        = {} # keep track of newly formed groups
        indices_          = []
        for i, pair in enumerate(self._pairs):
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

    def __getitem__(self, key):
        """Allows indexing of a ``PairList`` instance"""
        assert type(key) == int
        return self._pairs[key]

    def log(self):
        """Logs info from all pairs"""
        logger.info('{} PAIRS:'.format(len(self._pairs)))
        for pair in self._pairs:
            pair.log()

    def add_scores(self, scores):
        """Adds a score to each of the pairs"""
        assert len(scores) == self.npairs
        assert np.all(scores > 0)
        for i in range(len(scores)):
            self._pairs[i].score = scores[i]

    def sort(self):
        """Sorts the pairlist in-place according to assigned scores

        A ValueError is raised when a pair is encoutered without score.

        """
        for pair in self._pairs:
            assert pair.score is not None
        self._pairs.sort(key=lambda pair: pair.score) # use pair score as key

    def filter_scores(self, threshold, max_pairs=None):
        """Removes pairs for which the score falls above the threshold"""
        self.sort()
        assert threshold > self._pairs[0].score # threshold > minimum score
        index = 0
        while (index < self.npairs) and (self._pairs[index].score <= threshold):
            index += 1
        if max_pairs is None:
            # modify remove pairs above index
            self._pairs = self._pairs[:index]
        else:
            # only include subset of pairs
            self._pairs = self._pairs[:min(index, max_pairs)]
        self.npairs = len(self._pairs)

    def filter_disjunct(self):
        """Removes pairs until no cluster index appears more than once"""
        cluster_indices = []
        pairs_to_delete = []
        for i, pair in enumerate(self._pairs):
            if (pair._i in cluster_indices) or (pair._j in cluster_indices):
                # at least one of its indices is already present; exlude pair 
                pairs_to_delete.append(i)
            else:
                cluster_indices.append(pair._i)
                cluster_indices.append(pair._j)

        # delete pairs in-place and update npairs
        pairs_to_delete.sort(reverse=True)
        for i in pairs_to_delete: # delete higher indices first
            del self._pairs[i]
        self.npairs = len(self._pairs)

    def filter_equivalent(self, pair_reference, tol_distance=1e-3):
        """Retain only those pairs which are equivalent to a reference pair

        Arguments
        ---------

        pair_reference (``Pair`` instance):
            reference pair to which all pairs in self._pairs will be compared

        tol_distance (float):
            defines the tolerance for the distance
        """
        assert pair_reference in self
        pairs_to_delete = []
        for i, pair in enumerate(self._pairs):
            if not Pair.test_equivalence(pair_reference, pair, tol_distance):
                pairs_to_delete.append(i)

        # delete pairs in-place and update npairs
        pairs_to_delete.sort(reverse=True)
        for i in pairs_to_delete: # delete higher indices first
            del self._pairs[i]
        self.npairs = len(self._pairs)

    def __iter__(self):
        yield from self._pairs
