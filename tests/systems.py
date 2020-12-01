import ase

from pathlib import Path


# UIO-66
def _get_uio66():
    path_system = Path.cwd() / 'uio66'
    atoms = ase.io.read(path_system / 'conventional.cif')
    return atoms


def get_system(name):
    if name == 'uio66':
        return _get_uio66()
    else:
        raise NotImplementedError
