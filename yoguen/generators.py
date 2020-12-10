import logging
import numpy as np

from abc import ABC, abstractmethod
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from yoguen.candidates import Pair


logger = logging.getLogger(__name__) # logging per module


class Generator(ABC):
    """Abstract class for generators"""
    candidate_cls = None

    @abstractmethod
    def compute_candidates(self, clustering):
        """Generates and returns a list of ``Candidate`` instances"""
        pass

    @abstractmethod
    def select(self, clist, scores, tol_score=None):
        """Selects candidates based on their scores"""
        pass


class PairGenerator(Generator):
    """Generates a list of ``Pair`` instances"""
    candidate_cls = Pair

    def __init__(self, cutoff, max_neighbors, tol_distance=1e-2):
        assert type(max_neighbors) == int
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors
        self.tol_distance  = tol_distance

    def compute_candidates(self, clustering):
        """Generates and returns a list of ``Pair`` instances"""
        nlist = NeighborList(
                self.cutoff * np.ones(clustering.get_ncluster()) / 2,
                sorted=False,
                self_interaction=False,
                bothways=True,
                skin=0.0,
                primitive=NewPrimitiveNeighborList,
                )
        atoms_reduced = clustering.get_atoms_reduced()
        nlist.update(atoms_reduced) # construct neighborlist
        pairs = []
        for i in range(len(atoms_reduced)):
            neighbors, _ = nlist.get_neighbors(i) # contains duplicates
            distances = np.array([atoms_reduced.get_distance(i, a, mic=True) for a in list(neighbors)])
            sorting = distances.argsort()
            # start from nearest neighbors and add up to self.max_neighbors,
            # but avoid duplicates
            npairs_to_add = min(len(neighbors), self.max_neighbors)
            npairs_added  = 0
            for j in range(len(sorting)):
                pair = [i, neighbors[sorting[j]]]
                pair.sort()
                if npairs_added < npairs_to_add:
                    if tuple(pair) not in pairs:
                        pairs.append(tuple(pair))
                        npairs_added += 1
        # generate candidate list (clist) based on pairs
        clist = []
        for pair in pairs:
            clist.append(Pair.get_pair(clustering, *pair))
        return clist

    def select(self, clist, scores, tol_score):
        """selects candidates based on their scores

        Arguments
        ---------

        clist (list of ``Candidate`` subclass instances):
            list of candidates

        scores (1darray of length len(clist)):
            array of positive score values

        tol_score (None or double):
            candidates may only be selected if their score satisfies:
            (1 <=) score / min_score < tol_score.
            If this is None, then only the candidate with the minimal score
            will be selected.

        """
        # keep track of order
        id_list = [id(candidate) for candidate in clist]
        assert np.all(scores > 0) # scores should be strictly positive
        # sort candidates in clist according to their score
        scores_ = list(scores)
        sorted_zipped = sorted(zip(scores_, clist), key=lambda x: x[0])
        candidates_sorted = [candidate for _, candidate in sorted_zipped]
        scores_sorted = np.array([score for score, _ in sorted_zipped])
        # double check sorting
        # rebuild id list and verify clist order did not change
        assert np.allclose(np.sort(scores_sorted), scores_sorted)
        id_list_ = [id(candidate) for candidate in clist]
        assert np.all(np.array(id_list) == np.array(id_list_))

        # extract candidates whose score falls within the tolerance
        normalized_scores = np.sort(scores) / np.min(scores)
        qualified_candidates = np.where(normalized_scores - 1 <= tol_score)[0]

        # retain only those candidates that are equivalent with the winner
        winner = candidates_sorted[qualified_candidates[0]]
        selection = [winner]
        _within_tol = []
        for i in range(1, len(qualified_candidates)):
            equiv = Pair.possibly_equivalent(
                    winner,
                    candidates_sorted[qualified_candidates[i]],
                    tol_distance=self.tol_distance,
                    )
            if equiv:
                selection.append(candidates_sorted[qualified_candidates[i]])
            else:
                _within_tol.append(candidates_sorted[qualified_candidates[i]])
        return selection, _within_tol

    def filter_overlapping_pairs(self, clist):
        """Removes candidates until no two candidates share an atom"""
        clist_filtered = []
        pair_indices = []
        for pair in clist:
            non_overlapping = True
            for pair_tuple in pair_indices:
                if (pair[0] in pair_tuple) or (pair[1] in pair_tuple):
                    non_overlapping = False
            if non_overlapping:
                pair_indices.append(tuple([pair[0], pair[1]]))
                clist_filtered.append(pair)

        difference = len(clist) - len(clist_filtered)
        if difference > 0:
            logger.info('removed {} pairs due to overlapping atoms'.format(
                difference))
        return clist_filtered
