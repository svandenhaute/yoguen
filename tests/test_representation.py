import numpy as np
import automap
from pathlib import Path

from systems import get_system


def test_basis(tmp_path):
    atoms = get_system('uio66')
    ndof = 3 * len(atoms)
    prep = automap.PeriodicBasis(
            atoms,
            np.zeros((ndof, ndof)),
            )
