import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from automap.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum, get_logger
from automap.clustering import Clustering


logger = get_logger(__name__)


class GreedyReduction(object):
    """Represents the greedy reduction algorithm"""

    def __init__(self, cutoff, max_neighbors, ncluster_thres, temperature=300):
        """Constructor"""
        self.cutoff         = cutoff
        self.max_neighbors  = max_neighbors
        self.ncluster_thres = ncluster_thres
        self.temperature    = temperature

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
                pair = [i, neighbors[sorting[j]]]
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

        niter = 0  # tracks number of iterations
        smap  = [] # tracks smap for each pair

        logger.info('test')

        while not self.converged(clustering):
            print('')
            print('')
            print('-' * 10 + 'ITERATION {}'.format(niter) + '-' * 10)
            # compute list of candidate cluster pairs and compute smap array
            print('building pair list...')
            pairs = self.generate_pairs(clustering)
            print('selected {} pairs to evaluate'.format(len(pairs)))
            smap  = clustering._score_pairs(
                    quadratic,
                    pairs,
                    self.temperature,
                    )
            index = np.argmin(smap)
            print('found {} optimal at Smap = {} kJ/molK'.format(
                pairs[index],
                smap[index],
                ))
            atoms_reduced = clustering.get_atoms_reduced()
            symbols    = [None, None]
            symbols[0] = atoms_reduced.symbols[pairs[index][0]]
            symbols[1] = atoms_reduced.symbols[pairs[index][1]]
            distance = atoms_reduced.get_distance(
                    pairs[index][0],
                    pairs[index][1],
                    mic=True,
                    )
            print('with types {} at distance {} angstrom'.format(
                symbols,
                distance,
                ))

            # apply clustering and print extra info
            indices = clustering._join_pair(
                    clustering.get_indices(),
                    pairs[index],
                    )
            clustering.update_indices(indices)
            niter += 1


    def converged(self, clustering):
        """Determine whether or not the current clustering is sufficient"""
        return self.ncluster_thres >= clustering.get_ncluster()
