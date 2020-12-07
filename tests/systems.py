import yaml
import ase.io
import numpy as np

from pathlib import Path


# UIO-66
def _get_uio66():
    path_system = Path.cwd() / 'uio66'
    atoms = ase.io.read(path_system / 'conventional.cif')
    hessian  = np.load(path_system / 'hessian.npy')
    geometry = np.load(path_system / 'geometry.npy')
    cell = np.load(path_system / 'cell.npy')
    clusters = np.load(path_system / 'clusters.npy')

    yaml_dict = yaml.safe_load(open(path_system / 'clustering.yaml', 'rb'))
    indices_list = yaml_dict['indices']
    for i in range(len(indices_list)):
        indices_list[i] = tuple(indices_list[i])
    indices= tuple(indices_list)
    system = {
            'atoms': atoms,
            'geometry': geometry,
            'cell': cell,
            'hessian': hessian,
            'indices': indices,
            'clusters': clusters,
            }
    return system


def get_system(name):
    if name == 'uio66':
        return _get_uio66()
    else:
        raise NotImplementedError
