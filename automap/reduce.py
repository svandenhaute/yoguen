import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from automap.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum
from automap.clustering import Clustering


class GreedyReduction(object):
    """Represents the greedy reduction algorithm"""

    def __init__(self, cutoff, max_neighbors, ndof_thres):
        """Constructor"""
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors
        self.ndof_thres    = ndof_thres

    def generate_pairs(self, clustering):
        """Generates pairs of atoms that are close to each other"""
        nlist = NeighborList(
                self.cutoff * np.ones(clustering.get_ncluster()) / 2,
                sorted=False,
                self_interaction=False,
                bothways=False,
                skin=0.0,
                primitive=NewPrimitiveNeighborList,
                )
        # create reduced atoms object
        atoms_reduced = clustering.get_atoms_reduced()
        nlist.update(atoms_reduced)
        pairs = []
        for i in range(len(atoms_reduced)):
            neighbors, _ = nlist.get_neighbors(i)
            distances = np.array([atoms_reduced.get_distance(i, a, mic=True) for a in list(neighbors)])
            sorting = distances.argsort()
            for j in range(min(len(neighbors), self.max_neighbors)):
                pair = [i, sorting[j]]
                pair.sort()
                pairs.append(tuple(pair))
        return pairs

    def __call__(self, quadratic, clustering=None):
        """Applies the greedy reduction to a ``Quadratic`` instance

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            quadratic which should be reduced.

        clustering (``Clustering`` instance, optional):
            clustering from which the algorithm should start.

        """
        if clustering is None:
            clustering = Clustering(quadratic.atoms)

        # compute list of candidate cluster pairs
        pairs = self.generate_pairs(clustering)

        # list of indices
        indices_ = clustering.get_indices()
        symbols = list(clustering.get_atoms_reduced().symbols)
        for pair in pairs:
            indices = list(indices_)
            group = indices.pop(pair[1])
            indices[pair[0]] = indices[pair[0]] + group
            clustering.update_indices(tuple(indices))

    def converged(self, clustering):
        """Determine whether or not the current clustering is sufficient"""
        return self.ndof_thres > clustering.get_ncluster() * 3

