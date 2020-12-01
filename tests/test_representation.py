import automap
from pathlib import Path

from systems import get_system


def test_representation_basic(tmp_path):
    atoms = get_system('uio66')
    prep = automap.PeriodicRepresentation(atoms)
