import logging
import numpy as np

from yoguen.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum
from yoguen.clustering import Clustering
from yoguen.pairlist import PairList


logger = logging.getLogger(__name__) # logging per module


class GreedyReducer(object):
    """Represents the greedy reduction algorithm"""

    def __init__(self, cutoff, max_neighbors, temperature=300, verbose=True,
            tol_score=None, tol_distance=5e-3):
        """Constructor"""
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors
        self.temperature   = temperature
        self.verbose       = verbose
        self.tol_score     = tol_score
        self.tol_distance  = tol_distance

    def __call__(self, quadratic, max_ncluster, clustering=None,
            path_output=None):
        """Applies the greedy reduction to a ``Quadratic`` instance

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            quadratic which should be reduced.

        max_ncluster (int):
            specifies the number of clusters below which the reduction is
            finished.

        clustering (``Clustering`` instance, optional):
            initial clustering to start from

        path_output (``pathlib.Path`` instance, optional):
            directory to store output .pdb files to visualize progress

        """
        if clustering is None:
            clustering = Clustering(quadratic.atoms)
        else:
            assert id(clustering.atoms) == id(quadratic.atoms)
        niter = 0  # tracks number of iterations
        smap  = [] # tracks smap for each pair
        logger.info('')
        while max_ncluster < clustering.get_ncluster():
            logger.info('=' * 20 + '  ITERATION {}  '.format(niter) + '=' * 20)
            logger.info('current clustering:  {} atoms  --->  {} clusters'
                    ''.format(len(clustering.atoms), clustering.get_ncluster()))

            logger.info('building pair list...')
            pairlist = PairList.generate(
                    clustering,
                    cutoff=self.cutoff,
                    max_neighbors=self.max_neighbors,
                    )
            assert pairlist.npairs > 0
            logger.info('obtained {} candidates to evaluate'.format(
                pairlist.npairs))

            scores = clustering.score_pairlist(
                    pairlist,
                    quadratic,
                    self.temperature,
                    progress=self.verbose, # display progress if verbose
                    )
            # add scores to pairlist and sort accordingly
            pairlist.add_scores(scores)
            pairlist.sort()
            if self.tol_score is None:
                logger.warning('WARNING: inequivalent candidates not allowed')
                pairlist.filter_scores(np.min(scores), max_pairs=1) # retain 1
                assert pairlist.npairs == 1
            else:
                # remove pairs with score above threshold
                threshold = np.min(scores) * (1 + self.tol_score)
                pairlist.filter_scores(threshold)
            logger.info('retained {} pairs with score below {}'.format(
                pairlist.npairs,
                threshold,
                ))

            # only retain pairs equivalent with first one
            pairlist.filter_equivalent(pairlist[0], self.tol_distance)
            logger.info('retained {} equivalent pairs'.format(pairlist.npairs))
            pairlist.filter_disjunct() # remove pairs with overlapping indices
            logger.info('retained {} disjunct pairs'.format(pairlist.npairs))
            pairlist.log()
            indices = pairlist.apply( # apply pairlist to clustering
                    clustering.indices,
                    )
            clustering.update_indices(indices)

            if path_output is not None:
                clustering.visualize(
                        path_output / ('clustering_' + str(niter) + '.pdb'),
                        )
                clustering.save_indices(
                        path_output / ('indices_' + str(niter) + '.p'),
                        )
            niter += 1
            logger.info('')
            logger.info('')
