import automap

from systems import get_system


def test_greedy_reduce_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)

    greedy_reduce = automap.GreedyReduction(
            cutoff=3,
            max_neighbors=1,
            ndof_thres=84,
            )
    #greedy_reduce(quadratic)
