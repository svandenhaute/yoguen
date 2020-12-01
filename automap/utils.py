import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList


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
