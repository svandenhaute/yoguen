import numpy as np

from ase.units import kJ, mol

from automap.utils import get_mass_matrix, get_internal_basis, \
        compute_entropy_quantum, compute_entropy_classical


class Quadratic(object):
    """Class to represent a quadratic approximation"""
    allowed_kinds = ['plain', 'reduced', 'mw']

    def __init__(self, atoms, hessian, geometry, cell=None):
        """Constructor

        Arguments
        ---------

        atoms (``Atoms`` instance):
            Atoms object for which the quadratic approximation will be
            constructed.

        hessian (ndarray of shape (3 * natom, 3 * natom)):
            hessian matrix containing the second-order partial derivatives of
            the PES with respect to each of the atomic coordinates. No extended
            hessians are supported for the moment.
            in units of eV/(mol * angstrom ** 2)

        geometry (ndarray of shape (natom, 3)):
            equilibrium atomic geometry around which the quadratic is developed
            in angstrom

        cell (None or ndarray of shape (3, 3)):
            if the system is periodic, this argument specifies the box vectors
            that correspond to the equilibrium geometry.
            in angstrom

        """
        self.ndof = 3 * len(atoms)
        self.atoms = atoms

        assert hessian.shape == (self.ndof, self.ndof)
        assert np.allclose(hessian, hessian.T) # hessian should be symmetric
        self.hessian = hessian.copy()
        self.geometry = geometry.copy()
        if self.atoms.get_pbc().any():
            assert cell is not None
            self.cell = cell.copy()
        else:
            self.cell = None
            raise NotImplementedError

        ############################################################
        # ALLOCATE ARRAYS TO STORE EIGENVECTORS AND EIGENVALUES
        # 
        # plain:
        #   array of all eigenvectors of the plain hessian matrix on its
        #   columns
        # 
        # mw:
        #   array of all eigenvectors of the mass-weighted hessian
        #   matrix.
        #
        # reduced:
        #   array of subset of eigenvectors of the mass-weighted
        #   hessian; the excluded eigenvectors are those belonging to
        #   global translations (and rotations in case of nonperiodic
        #   systems).
        ############################################################
        self.modes  = {}
        self.values = {}
        self.modes['plain']  = np.zeros((self.ndof, self.ndof))
        self.values['plain'] = np.zeros(self.ndof)
        self.modes['mw']     = np.zeros((self.ndof, self.ndof))
        self.values['mw']    = np.zeros(self.ndof)
        if self.cell is not None:
            self.modes['reduced']  = np.zeros((self.ndof - 3, self.ndof - 3))
            self.values['reduced'] = np.zeros(self.ndof - 3)

    def get_modes_values(self, kind='reduced'):
        """Returns the eigenmodes and eigenvalues

        Arguments
        ---------

        kind (str):
            specifies the kind of modes/values that are expected.

        """
        assert kind in self.allowed_kinds
        if not self.values[kind].any():
            self._compute_modes_values(kind)
        return self.modes[kind].copy(), self.values[kind].copy()

    def get_conversion(self, kind='reduced'):
        assert kind in self.allowed_kinds
        if kind == 'plain':
            return np.eye(self.ndof)
        elif kind == 'mw':
            mm = get_mass_matrix(self.atoms)
            return np.sqrt(mm)
        elif kind == 'reduced':
            basis = get_internal_basis(self.atoms, mw=True)
            return basis

    def _compute_modes_values(self, kind):
        """Computes eigenmodes and eigenvalues"""
        if kind == 'plain': # compute plain eigenmodes and eigenvalues
            w, v = np.linalg.eigh(self.hessian)
            v.sort()
        elif kind == 'mw': # compute mass-weighted
            _ = np.sqrt(get_mass_matrix(self.atoms))
            hessian_m = np.linalg.inv(_) @ self.hessian @ np.linalg.inv(_)
            w, v = np.linalg.eigh(hessian_m) # apply mass-weighting
            v.sort()
        elif kind == 'reduced': # compute reduced
            _ = np.sqrt(get_mass_matrix(self.atoms))
            hessian_m = np.linalg.inv(_) @ self.hessian @ np.linalg.inv(_)
            basis = get_internal_basis(self.atoms, mw=True)
            hessian_r = basis.T @ hessian_m @ basis
            w, v = np.linalg.eigh(hessian_r)
            v.sort()
        self.modes[kind][:] = v
        self.values[kind][:] = w

    def compute_entropy(self, T=300, quantum=True):
        """Computes the (quantum) entropy based on the hessian eigenvalues.

        The entropy is returned in kJ/(mol K).

        Arguments
        ---------

        T (double):
            temperature in kelvin.

        """
        _, omegas2 = self.get_modes_values(kind='reduced')
        if quantum:
            return np.sum(compute_entropy_quantum(np.sqrt(omegas2) / (2 * np.pi), T))
        else:
            return np.sum(compute_entropy_classical(np.sqrt(omegas2) / (2 * np.pi), T))
