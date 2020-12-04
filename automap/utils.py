import molmod
import numpy as np

from ase.units import create_units
#from ase.neighborlist import NeighborList, NewPrimitiveNeighborList


UNIT_INVCM = 7.251632778591094e-07


def get_units():
    u = create_units('2014')
    return u


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


def compute_entropy_quantum(f, T):
    u = get_units()
    if (f > 0).all():
        h = u._hplanck # in J s
        k = u._k # in J / K
        f_si = f / u._aut # frequencies in Hz
        beta = 1 / (k * T)
        thetas = beta * h * f_si # dimensionless
        q_quantum = np.exp(- thetas / 2) / (1 - np.exp(- thetas))
        f_quantum = - np.log(q_quantum) / beta
        s_quantum = -k * (np.log(1 - np.exp(- thetas)) - thetas * (np.exp(thetas) - 1) ** (-1))
        s_quantum /= 1000
        s_quantum *= u._Nav
        return s_quantum
    else:
        raise ValueError('Entropy at 0Hz is infinite')


def get_cluster_positions(atoms, clustering):
    """Computes the positions of the clusters"""
    ncluster = clustering.get_ncluster()
    indices  = clustering.get_indices()
    pos_c    = np.zeros((ncluster, 3))
    pos      = atoms.get_positions()
    masses   = atoms.get_masses()

    for i, group in enumerate(indices):
        # compute total mass of group
        mass = np.sum(masses[np.array(group)])

        # first atom is used as reference. COM is computed using relative
        # vectors only, in order to apply the mic consistently 
        index_ref = group[0]
        pos_c[i, :] = pos[index_ref, :]
        for i in range(1, len(group)):
            index = group[i]
            delta = atoms.get_distance(index_ref, index, mic=True, vector=True)
            pos_c[i, :] += masses[index] / mass * delta
    return pos_c


def get_cluster_elements(atoms, clustering):
    """Returns elements of clusters

    Clusters that are chemically equivalent should have the same element.
    Clusters containing only one atom should have the element of that atom.
    Cluster elements start at 118 and count backward.

    """
    ncluster  = clustering.get_ncluster()
    indices   = clustering.get_indices()
    numbers   = atoms.get_atomic_numbers()
    numbers_c = np.zeros(ncluster)

    cluster_elements = {}
    new_key = 118

    for i, group in enumerate(indices):
        if len(group) == 1: # if only one atom, then number is same
            numbers_c[i] = numbers[group[0]]
        else: # if multiple atoms, then start at 118
            numbers_in_group = set(numbers[np.array(group)])
            for key, value in cluster_elements.items():
                if value == numbers_in_group:
                    numbers_c[i] = key
                else:
                    cluster_elements[new_key] = numbers_in_group
                    new_key -= 1
    return numbers_c


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
