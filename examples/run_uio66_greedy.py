import yaml
import logging
import yoguen
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
    indices  = system['indices']

    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    greducer  = yoguen.GreedyReducer(
            cutoff=5,
            max_neighbors=3,
            temperature=300,
            verbose=True,
            tol_score=1e-2,
            tol_distance=1e-2,
            )
    greducer(quadratic, 28, path_output=Path.cwd())
