import numpy as np

from ase.units import create_units
#from ase.neighborlist import NeighborList, NewPrimitiveNeighborList


UNIT_INVCM = 7.251632778591094e-07


def get_units():
    u = create_units('2014')
    return u


def get_reduced_basis(atoms, mw=False):
    """Creates basis for reduced coordinates

    If the system is periodic, then only three global translations are removed.
    If the system is aperiodic, then additionally three global rotations are
    removed.

    Arguments
    ---------

    atoms (``Atoms`` instance):
        Atoms instance of the system

    mw (bool):
        whether or not to construct the reduced basis for mass-weighted
        coordinates.

    """
    if atoms.get_pbc().any():
        t = np.zeros((3, len(atoms) * 3))
        t[0, 0::3] = 1
        t[1, 1::3] = 1
        t[2, 2::3] = 1
        if mw: # apply mass weighting
            masses = atoms.get_masses()
            u = get_units()
            masses *= (u._amu / u._me) # convert to atomic units
            t_mw = t @ np.sqrt(np.diag(np.repeat(masses, 3)))
        _, _, vH = np.linalg.svd(t_mw)
        basis = np.transpose(vH[3:])
        return basis
    else:
        raise NotImplementedError


def get_mass_matrix(atoms):
    """Returns the mass matrix

    Arguments
    ---------

    atoms (``Atoms`` instance):
        provides the atomic masses. The returned mass matrix is square with
        dimension 3 * len(atoms).

    """
    weights = np.zeros((3 * len(atoms), 3 * len(atoms)))
    masses = atoms.get_masses()
    u = get_units()
    masses *= (u._amu / u._me) # convert to atomic units
    weights[:] = np.diag(np.repeat(masses, 3))
    #for i in range(len(atoms)):
    #    for j in range(len(atoms)):
    #        weights[i][j] = 1 / np.sqrt(masses[i] * masses[j])
    return weights


def _entropy_quantum(f, T):
    if (f > 0).all():
        h = molmod.constants.planck
        k = molmod.constants.boltzmann
        beta = 1 / (molmod.constants.boltzmann * T)
        q_quantum = np.exp(- (beta * h * f) / 2) / (1 - np.exp(- beta * h * f))
        f_quantum = - np.log(q_quantum) / beta
        s_quantum = -k * np.log(1 - np.exp(- beta * h * f)) + h * f / T * (np.exp(beta * h * f) - 1) ** (-1)
        return s_quantum
    else:
        raise ValueError('Entropy at 0Hz is infinite')


def _entropy_classical(f, T=300):
    if (f > 0).all():
        h = molmod.constants.planck
        k = molmod.constants.boltzmann
        beta = 1 / (molmod.constants.boltzmann * T)
        q_classical = 1 / (beta * h * f)
        f_classical = - np.log(q_classical) / beta
        s_classical = k * (1 + np.log(k * T / (h * f)))
        return s_classical
    else:
        raise ValueError('Entropy at 0Hz is infinite')


#def infer_bonds(atoms, mic=True, thresh=2.5):
#    """determines the bonds based on interatomic distances
#
#    Arguments
#    ---------
#
#    atoms (``Atoms`` instance):
#        represents the molecular system for which bonds should be inferred.
#
#    mic (bool):
#        determines whether or not to use the minimum image convention to
#        determine interatomic distances (in case of periodic systems).
#
#    thresh (float):
#        largest interatomic distance between atoms that are considered bonded;
#        in angstrom.
#
#    """
#    bonds = []
#    natom = len(atoms)
#    cutoffs = np.zeros(natom)
#    for i in range(natom):
#        cutoffs[i] = covalent_radii[atoms.numbers[i]]
#    print(cutoffs)
#    # create list of cutoffs
#    nlist = NeighborList(
#            cutoffs,
#            sorted=True,
#            self_interaction=False,
#            primitive=NewPrimitiveNeighborList,
#            )
#    nlist.update(atoms)
#    for i in range(natom):
#        neighbors, _ = nlist.get_neighbors(i)
#        print(atoms.symbols[i], i, neighbors)
#        assert len(neighbors) > 0 # all atoms have at least one bond
#        for n in neighbors:
#            print(atoms.get_distance(i, n, mic=True))
#            bonds.append((i, n))
#    return bonds
