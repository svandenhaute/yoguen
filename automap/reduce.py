import logging
import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from automap.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum
from automap.clustering import Clustering


logger = logging.getLogger(__name__) # logging per module


class GreedyReduction(object):
    """Represents the greedy reduction algorithm"""

    def __init__(self, cutoff, max_neighbors, ncluster_thres, temperature=300,
            path_output=None):
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
        return pairs[:20]

    def __call__(self, quadratic, clustering=None, path_output=None,
            progress=True):
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

        logger.critical('test')

        while not self.converged(clustering):
            logger.info('')
            logger.info('')
            logger.info('-' * 10 + 'ITERATION {}'.format(niter) + '-' * 10)
            # compute list of candidate cluster pairs and compute smap array
            logger.info('building pair list...')
            pairs = self.generate_pairs(clustering)
            logger.info('selected {} pairs to evaluate'.format(len(pairs)))
            scores  = clustering._score_pairs(
                    quadratic,
                    pairs,
                    self.temperature,
                    progress=progress,
                    )
            index = np.argmin(scores) # get pair with minimal Smap
            self.report(
                    clustering,
                    pairs[index],
                    scores[index],
                    smap,
                    )
            indices = clustering._join_pair( # join clusters
                    clustering.get_indices(),
                    pairs[index],
                    )
            clustering.update_indices(indices) # apply clustering
            smap.append(scores[index])

            if path_output is not None:
                clustering.visualize(
                        path_output / ('clustering_' + str(niter) + '.pdb'),
                        )
            niter += 1

    def report(self, clustering, pair, score, smap):
        """Reports result of pair scoring"""
        atoms_reduced = clustering.get_atoms_reduced()
        symbols    = [None, None]
        symbols[0] = atoms_reduced.symbols[pair[0]]
        symbols[1] = atoms_reduced.symbols[pair[1]]
        distance = atoms_reduced.get_distance(
                pair[0],
                pair[1],
                mic=True,
                )
        indices = clustering.get_indices()
        groups = [indices[pair[0]], indices[pair[1]]]
        symbols0 = clustering.get_elements_in_cluster(pair[0])
        symbols1 = clustering.get_elements_in_cluster(pair[1])
        # define entropies
        total = score
        if len(smap) == 0:
            increment = score
        else:
            increment = score - smap[-1]
        logger.info('selected pair {}'.format(pair))
        logger.info('\ttypes {} and {}'.format(symbols[0], symbols[1]))
        logger.info('\tdistance {} angstrom'.format(distance))
        logger.info('\tjoining atomic indices {} and {}'.format(groups[0], groups[1]))
        logger.info('\twith atomic elements {} and {}'.format(symbols0, symbols1))
        logger.info('entropy increment of   {:.7e} kJ/molK'.format(increment))
        logger.info('total mapping entropy  {:.7e} kJ/molK'.format(total))

    def converged(self, clustering):
        """Determine whether or not the current clustering is sufficient"""
        return self.ncluster_thres >= clustering.get_ncluster()
