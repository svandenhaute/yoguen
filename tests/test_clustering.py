import automap
import numpy as np

from systems import get_system


def test_score_pairs(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']
    clusters = system['clusters']

    clustering = automap.Clustering(atoms)
    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)
    pairs = [ # random atom indices in [0, 455]
            (0, 1),
            (234, 13),
            (67, 350),
            (6, 450),
            (0, 1),
            ]
    smap = clustering._score_pairs(quadratic, pairs, progress=False)

    # do manual calculation using apply()
    smap_manual = np.zeros(len(pairs))
    indices = clustering.get_indices()
    for i, pair in enumerate(pairs):
        _indices = clustering._join_pair(indices, pair)
        clustering.update_indices(_indices)
        entropies, _ = clustering.apply(quadratic)
        smap_manual[i] = entropies[1]
        clustering.update_indices(indices) # revert back to default clustering
    assert np.allclose(smap, smap_manual)

    # test fast version
    smap_fast = clustering._score_pairs_fast(quadratic, pairs, progress=False)
    assert np.allclose(smap, smap_fast)


def test_clustering_uio66(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    clustering = automap.Clustering(atoms)
    clustering.update_indices(indices) # set non-trivial mapping

    #clustering.complete_clusters_by_translation()
    #clustering.atoms.write('/home/sandervandenhaute/conventional.cif')
    #clustering.visualize('/home/sandervandenhaute/clustering.pdb')
    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)
    entropies, quadratic_reduced = clustering.apply(quadratic)
    assert entropies[2] < quadratic.compute_entropy() # verify entropy decrease


def test_clustering_basic(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    clustering = automap.Clustering(atoms)
    assert clustering.get_ncluster() == len(atoms)
    assert clustering.validate()


def test_clustering_compute_loss(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    clustering = automap.Clustering(atoms)
    quadratic  = automap.Quadratic(atoms, hessian, geometry, cell)
    saa = quadratic.compute_entropy(300)

    # compute loss for identical mapping --> should be zero
    entropies, quadratic_reduced = clustering.apply(quadratic)
    assert abs(saa - entropies[0]) < 1e-8


def test_get_atoms_reduced(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']
    clusters = system['clusters']
    atoms.set_positions(geometry) # set atoms to minimum energy configuration
    atoms.set_cell(cell)

    clustering = automap.Clustering(atoms)
    clustering.update_indices(indices)
    atoms_reduced = clustering.get_atoms_reduced()
    pos_reduced = atoms_reduced.get_positions()
    assert np.all(np.abs(pos_reduced - clusters) < 1e-4)
