import molmod
import logging
import numpy as np

import ase.units
#from ase.neighborlist import NeighborList, NewPrimitiveNeighborList


def get_logger(name, level=20):
    """Returns logger

    The default logging level is logging.INFO (20).

    """
    logging.basicConfig(
            format='%(name) - %(message)',
            level=20,
            )
    return logging.getLogger(name)


def get_internal_basis(atoms, mw=False):
    """Creates basis for internal coordinates

    If the system is periodic, then only three global translations are removed.
    If the system is aperiodic, then additionally three global rotations are
    removed.

    Arguments
    ---------

    atoms (``Atoms`` instance):
        Atoms instance of the system

    mw (bool):
        whether or not to construct the internal basis for mass-weighted
        coordinates.

    """
    if atoms.get_pbc().any():
        t = np.zeros((3, len(atoms) * 3))
        t[0, 0::3] = 1
        t[1, 1::3] = 1
        t[2, 2::3] = 1
        if mw: # apply mass weighting
            masses = atoms.get_masses()
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
    weights[:] = np.diag(np.repeat(masses, 3))
    return weights


def compute_entropy_quantum(f, T):
    if (f > 0).all():
        h = ase.units._hplanck # in J s
        k = ase.units._k # in J / K
        f_si = f * ase.units.s # frequencies in Hz
        beta = 1 / (k * T)
        thetas = beta * h * f_si # dimensionless
        q_quantum = np.exp(- thetas / 2) / (1 - np.exp(- thetas))
        f_quantum = - np.log(q_quantum) / beta
        s_quantum = -k * (np.log(1 - np.exp(- thetas)) - thetas * (np.exp(thetas) - 1) ** (-1))
        s_quantum /= 1000
        s_quantum *= ase.units._Nav # to kJ/mol
        return s_quantum
    else:
        raise ValueError('Entropy at 0Hz is infinite')


def compute_entropy_classical(f, T):
    if (f > 0).all():
        h = ase.units._hplanck # in J s
        k = ase.units._k # in J / K
        f_si = f * ase.units.s # frequencies in Hz
        beta = 1 / (k * T)
        thetas = beta * h * f_si # dimensionless
        q_classical = 1 / (thetas)
        f_classical = - np.log(q_classical) / beta
        s_classical = k * (1 + np.log(1 / thetas))
        s_classical /= 1000
        s_classical *= ase.units._Nav
        return s_classical
    else:
        raise ValueError('Entropy at 0Hz is infinite')


def expand_mapping(projection):
    """Expands a projection matrix to triple its size

    Clusters and projection arrays are usually stored in an array of size
    (natom, natom) because no distinction is made between the x, y, and z
    components. This function enlarges these arrays to size (3natom, 3natom)
    by row/column permutations on a diagonal block matrix of three copies.
    """
    N = projection.shape[0]
    n = projection.shape[1]
    block = np.kron(np.eye(3), projection)
    indices_n = np.arange(3 * n).reshape(3, n).T.flatten()
    indices_N = np.arange(3 * N).reshape(3, N).T.flatten()
    permute_rows = block[indices_N, :]
    total = permute_rows[:, indices_n]
    return total


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
