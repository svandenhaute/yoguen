import numpy as np
import automap
from pathlib import Path
from ase.units import invcm, s, _c

from systems import get_system


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

    # compute frequencies
    _, values = quad.get_modes_values('mw')
    assert len(values) == len(atoms) * 3
    frequencies = np.sqrt(np.abs(values)) / (2 * np.pi)
    frequencies_invcm = frequencies * s / _c / 100
    assert np.all(frequencies_invcm < 4e3) # all frequencies < 4000 invcm
    assert np.linalg.norm(frequencies[:3]) < 1e-1

    # compute frequencies without translational modes
    _, values = quad.get_modes_values('reduced')
    assert len(values) == len(atoms) * 3 - 3
    frequencies = np.sqrt(np.abs(values)) / (2 * np.pi) # in np.sqrt(eV / A**2)
    frequencies_invcm = frequencies * s / _c / 100
    #frequencies_hz = frequencies * s
    assert np.all(frequencies_invcm < 4e3) # all frequencies < 4000 invcm
    assert np.linalg.norm(frequencies_invcm[:3]) > 1e-1 # first 3 eigenvalues NONzero

    # compute entropy and check with reference value of 5.328 kJ/(mol K)
    assert abs(quad.compute_entropy(300) - 5.328) < 1e-2
