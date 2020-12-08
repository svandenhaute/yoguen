import yoguen
from pathlib import Path

from systems import get_system


def test_generate_pairs_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    #quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    clustering = yoguen.Clustering(atoms)
    greedy_reduce = yoguen.GreedyReduction(
            cutoff=5, # cutoff radius in angstrom
            max_neighbors=1, # starting from nearest neighbor
            ncluster_thres=455, # threshold for number of clusters
            )
    pairs = greedy_reduce.generate_pairs(clustering)

    # ensure pairs are sorted
    for pair in pairs:
        assert pair[0] < pair[1]

    # ensure pairs are unique
    pairs_ = list(pairs)
    while len(pairs_) > 0:
        pair = pairs_.pop(-1)
        assert pair not in pairs_


def test_greedy_reduce_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    path_output = Path.cwd()

    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)

    greedy_reduce = yoguen.GreedyReduction(
            cutoff=5, # cutoff radius in angstrom
            max_neighbors=1, # starting from nearest neighbor
            ncluster_thres=455, # threshold for number of clusters
            )
    greedy_reduce(
            quadratic,
            path_output=None,
            progress=False,
            )
