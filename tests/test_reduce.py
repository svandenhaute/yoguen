import yoguen
from pathlib import Path

from systems import get_system


def test_greedy_reduce_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    path_output = Path.cwd()

    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    generator = yoguen.PairGenerator(
            cutoff=5, # cutoff radius in angstrom
            max_neighbors=1, # starting from nearest neighbor
            )
    greducer = yoguen.GreedyReducer(
            generator,
            temperature=300, # temperature in kelvin
            verbose=True, # suppress output
            tol_score=1e-3, # group equivalent candidates
            )
    greducer(quadratic, len(atoms) - 1, path_output=None)
