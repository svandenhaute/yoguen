import automap
from pathlib import Path

from systems import get_system


def test_greedy_reduce_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    path_output = Path.cwd()

    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)

    greedy_reduce = automap.GreedyReduction(
            cutoff=5, # cutoff radius in angstrom
            max_neighbors=1, # starting from nearest neighbor
            ncluster_thres=455, # threshold for number of clusters
            )
    greedy_reduce(
            quadratic,
            path_output=None,
            progress=False,
            )
