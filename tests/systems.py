import ase.io
import numpy as np

from pathlib import Path


# UIO-66
def _get_uio66():
    path_system = Path.cwd() / 'uio66'
    atoms = ase.io.read(path_system / 'conventional.cif')
    hessian  = np.load(path_system / 'hessian_conventional.npy')
    geometry = np.load(path_system / 'geometry_conventional.npy')
    cell = np.load(path_system / 'cell_conventional.npy')
    return atoms, (geometry, cell, hessian)


def get_system(name):
    if name == 'uio66':
        return _get_uio66()
    else:
        raise NotImplementedError
