import yoguen
from pathlib import Path

from systems import get_system


def test_greedy_reduce_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    greducer  = yoguen.GreedyReducer(
            cutoff=3,
            max_neighbors=1,
            temperature=300,
            verbose=True,
            tol_score=1e-2,
            tol_distance=5e-3,
            )
    greducer(quadratic, len(atoms) - 1, path_output=Path.cwd())
