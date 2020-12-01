
def infer_bonds(atoms, mic=True, thresh=1.8):
    """determines the bonds based on interatomic distances

    Arguments
    ---------

    atoms (``Atoms`` instance):
        represents the molecular system for which bonds should be inferred.

    mic (bool):
        determines whether or not to use the minimum image convention to
        determine interatomic distances (in case of periodic systems).

    thresh (float):
        largest interatomic distance between atoms that are considered bonded;
        in angstrom.

    """
    bonds = []
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            if atoms.get_distance(i, j, mic=mic) < thresh:
                bonds.append((i, j))
    return bonds

