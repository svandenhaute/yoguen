import logging
import numpy as np

from yoguen.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum
from yoguen.clustering import Clustering


logger = logging.getLogger(__name__) # logging per module


class GreedyReducer(object):
    """Represents the greedy reduction algorithm"""

    def __init__(self, generator, temperature=300, verbose=True,
            tol_score=None):
        """Constructor"""
        self.generator   = generator
        self.temperature = temperature
        self.verbose     = verbose
        self.tol_score   = tol_score

    def __call__(self, quadratic, max_ncluster, path_output=None):
        """Applies the greedy reduction to a ``Quadratic`` instance

        Arguments
        ---------

        quadratic (``Quadratic`` instance):
            quadratic which should be reduced.

        max_ncluster (int):
            specifies the number of clusters below which the reduction is
            finished.

        path_output (``pathlib.Path`` instance):
            directory to store output .pdb files to visualize progress

        """
        clustering = Clustering(quadratic.atoms)
        niter = 0  # tracks number of iterations
        smap  = [] # tracks smap for each pair
        logger.info('')
        while not max_ncluster > clustering.get_ncluster():
            logger.info('=' * 20 + '  ITERATION {}  '.format(niter) + '=' * 20)
            logger.info('current clustering:  {} atoms  --->  {} clusters'
                    ''.format(len(clustering.atoms), clustering.get_ncluster()))

            logger.info('building pair list...')
            clist = self.generator.compute_candidates(clustering)

            logger.info('obtained {} candidates to evaluate'.format(len(clist)))
            scores = clustering.score_candidates(
                    clist,
                    quadratic,
                    self.temperature,
                    progress=self.verbose, # display progress if verbose
                    )
            selection, _within_tol = self.generator.select(
                    clist,
                    scores,
                    self.tol_score,
                    )
            if self.tol_score is None:
                logger.warning('WARNING: inequivalent candidates not allowed')
            logger.info('selected {} candidate(s):'.format(len(selection)))
            for candidate in selection:
                candidate.log()
            if len(_within_tol) > 0:
                logger.info('first EXCLUDED candidate:')
                _within_tol[0].log()

            # filter selection and update clustering
            selection_filtered = self.generator.filter_overlapping_pairs(
                    selection,
                    )
            indices = self.generator.candidate_cls.apply_clist(
                    selection_filtered,
                    clustering.indices,
                    )
            clustering.update_indices(indices)

            if path_output is not None:
                clustering.visualize(
                        path_output / ('clustering_' + str(niter) + '.pdb'),
                        )
            niter += 1
            logger.info('')
            logger.info('')

    def report(self, clustering, pair, score=None, smap=None):
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
        logger.info('selected pair {}'.format(pair))
        logger.info('\ttypes {} and {}'.format(symbols[0], symbols[1]))
        logger.info('\tdistance between clusters is {:.5e} angstrom'.format(distance))
        logger.info('\tjoining atomic indices {} and {}'.format(groups[0], groups[1]))
        logger.info('\twith atomic elements {} and {}'.format(symbols0, symbols1))
        if score is not None and smap is not None:
            total = score
            if len(smap) == 0:
                increment = score
            else:
                increment = score - smap[-1]
            logger.info('entropy increment:       {:.7e} kJ/molK'.format(increment))
            logger.info('total mapping entropy:   {:.7e} kJ/molK'.format(total))
