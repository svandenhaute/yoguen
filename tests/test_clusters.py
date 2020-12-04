import automap

from systems import get_system


def test_clustering_basic(tmp_path):
    # generate trivial clustering for UiO-66, verify output of helper functions
    atoms, _ = get_system('uio66')

    clustering = automap.Clustering(atoms)
    assert clustering.get_ncluster() == len(atoms)
    assert clustering.validate()
