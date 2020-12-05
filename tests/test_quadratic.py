import numpy as np
import automap
import molmod
from pathlib import Path

from systems import get_system


UNIT_INVCM = 7.251632778591094e-07


def test_quadratic(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    quad = automap.Quadratic(atoms, hessian, geometry, cell)

    # compute plain eigenvalues
    _, values = quad.get_modes_values('plain')
    assert len(values) == len(atoms) * 3
    frequencies = np.sqrt(np.abs(values)) / (2 * np.pi) / UNIT_INVCM
    assert np.mean(frequencies[:3]) < 1e-1 # first 3 frequencies essentially 0

    # compute frequencies
    _, values = quad.get_modes_values('mw')
    assert len(values) == len(atoms) * 3
    frequencies = np.sqrt(np.abs(values)) / (2 * np.pi) / UNIT_INVCM
    assert np.linalg.norm(frequencies[:3]) < 1e-1

    # compute frequencies without translational modes
    _, values = quad.get_modes_values('reduced')
    assert len(values) == len(atoms) * 3 - 3
    frequencies = np.sqrt(np.abs(values)) / (2 * np.pi) / UNIT_INVCM
    assert np.linalg.norm(frequencies[:3]) > 1e-1 # first 3 eigenvalues NONzero

    # compute entropy and check with reference value of 5.328 kJ/(mol K)
    assert abs(quad.compute_entropy(300) - 5.328) < 1e-2
