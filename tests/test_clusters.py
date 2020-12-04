import automap

from systems import get_system


def test_clustering_basic(tmp_path):
    # generate trivial clustering for UiO-66, verify output of helper functions
    atoms, (geometry, cell, hessian) = get_system('uio66')

    clustering = automap.Clustering(atoms)
    assert clustering.get_ncluster() == len(atoms)
    assert clustering.validate()


def test_clustering_compute_loss(tmp_path):
    atoms, (geometry, cell, hessian) = get_system('uio66')
    clustering = automap.Clustering(atoms)
    quadratic  = automap.Quadratic(atoms, hessian, geometry, cell)

    smap = automap.Reduction.compute_loss(clustering, quadratic, T=300)
    print(smap)
