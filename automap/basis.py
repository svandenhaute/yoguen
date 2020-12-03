from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy


class PeriodicBasis(object):
    """Represents a coordinate basis of a periodic molecular system"""

    def __init__(self, atoms, transform, fixed_cell=True):
        """Constructor

        Arguments
        ---------

        atoms (``Atoms`` instance):
            atoms instance for which to create the representation. Its current
            positions and cell vectors will be used as coordinate reference.

        transform (ndarray):
            provides the (mass-unweighted) transformation matrix between the
            atomic cartesian coordinates and the new basis. The shape of this
            transform is determined by the number of atoms in the system and
            the value of fixed_cell. If fixed_cell is True, then the shape
            should be (3 * natom, 3 * natom).

        fixed_cell (bool):
            determines whether the coordinates are included as variables or are
            assumed fixed.

        """
        self._reference = deepcopy(atoms)
        self.fixed_cell = fixed_cell

        # assert correct dimensions of transformation matrix
        natom = len(atoms)
        if fixed_cell:
            assert transform.shape == (3 * natom, 3 * natom)


    def apply(self, atoms):
        """Applies the transformation"""
        pass
