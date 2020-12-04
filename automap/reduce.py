import numpy as np
from abc import ABC, abstractmethod

from automap.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum


class Reduction(ABC):
    """Base class to execute a dimensionality reduction algorithm"""

    @abstractmethod
    def __call__(self, clustering, quadratic):
        pass

    @staticmethod
    def compute_loss(clustering, quadratic, T=300):
        """Computes the CG and mapping entropy for the clustering

        The steps are as follows:
            (0) start with the mass-weighted hessian and mapping transformation
                arrays
            (1) compute eigenmodes and eigenvalues of hessian
            (2) generate internal basis that removes global translations (and
                rotations in case of nonperiodic systems) and transform hessian
                and mapping transformation.
            (3) compute the SVD of the mapping
            (4) transform the internal hessian in the generalized row space of
                the mapping
            (5) diagonalize the lower hessian submatrix to obtain the mapping
                entropy.

        Arguments
        ---------

        clustering (``Clustering`` instance):
            instance representing the specific clustering of atoms for which
            the entropies must be computed.

        quadratic (``Quadratic`` instance):
            instance representing the local PES.

        T (double):
            temperature in Kelvin. This is required to compute the entropy
            of the individual harmonic oscillators.

        """
        atoms_reduced = clustering.get_atoms_reduced()

        # mass diagonal matrix for atomistic system
        W_r = get_mass_matrix(clustering.atoms)
        # internal basis for atomistic representation
        B_r = get_internal_basis(clustering.atoms, mw=True)

        # mass diagonal matrix for reduced system
        W_R = get_mass_matrix(atoms_reduced)
        # internal basis for reduced representation
        B_R = get_internal_basis(atoms_reduced, mw=True)

        # get arrays and apply mass-weighting
        mapping   = clustering.get_mapping()
        mapping_m = np.sqrt(W_R) @ mapping @ np.linalg.inv(np.sqrt(W_r))
        hessian   = quadratic.hessian.copy()
        hessian_m = np.linalg.inv(np.sqrt(W_r)) @ hessian @ np.linalg.inv(np.sqrt(W_r))

        # transform to internal coordinates
        mapping_ic = np.transpose(B_R) @ mapping_m @ B_r
        hessian_ic = np.transpose(B_r) @ hessian_m @ B_r
        _, sigmas, KNT = np.linalg.svd(mapping_ic)
        assert np.allclose(sigmas, np.ones(sigmas.shape)) # sing. values == 1
        KN = np.transpose(KNT) # generalized row space of mapping

        # transform hessian into generalized row space, create blocks
        hessian_row = np.tranpose(KN) @ hessian_ic @ KN
        size = mapping_ic.shape[0]
        hessian_11 = hessian_row[:size, :size]
        hessian_12 = hessian_row[:size, size:]
        hessian_22 = hessian_row[size:, size:]

        # diagonalize lower right block to obtain frequencies; compute entropy
        _, omegas = np.linalg.eigh(hessian_22)
        frequencies = np.sqrt(omegas) / (2 * np.pi)
        return np.sum(compute_entropy_quantum(frequencies, T))


class GreedyReduction(Reduction):
    """Represents the greedy reduction algorithm"""
    pass
