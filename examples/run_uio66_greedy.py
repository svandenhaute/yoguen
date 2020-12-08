import yaml
import logging
import automap
import ase.io
import numpy as np

from pathlib import Path


def get_system():
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


if __name__ == '__main__': # actual test
    # enable logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

    # get system files to create quadratic and clustering
    system = get_system()
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)
    greedy_reduce = automap.GreedyReduction(
            cutoff=5,
            max_neighbors=1, # starting from nearest neighbor
            ncluster_thres=455,
            )
    greedy_reduce(quadratic, progress=False)
