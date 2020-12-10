import yoguen
import numpy as np

from systems import get_system


def test_score_pairs(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    clustering = yoguen.Clustering(atoms)
    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    pair_indices = [ # random atom indices in [0, 455]
            (0, 1),
            (234, 13),
            (6, 450),
            (2, 3),
            ]
    clist = [yoguen.Pair.get_pair(clustering, *pair) for pair in pair_indices]
    smap = clustering.score_candidates(clist, quadratic, progress=False) # at 300 K

    # do manual calculation using apply()
    smap_manual = np.zeros(len(clist))
    indices = clustering.indices # keep track of original indices
    for i, pair in enumerate(clist):
        indices_ = pair.apply(indices)
        clustering.update_indices(indices_)
        entropies, _ = clustering.apply(quadratic)
        smap_manual[i] = entropies[1]
        clustering.update_indices(indices) # revert back to default clustering
    assert np.allclose(smap, smap_manual)


def test_clustering_uio66(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    clustering = yoguen.Clustering(atoms)
    clustering.update_indices(indices) # set non-trivial mapping
    quadratic = yoguen.Quadratic(atoms, hessian, geometry, cell)
    entropies, quadratic_reduced = clustering.apply(quadratic)
    assert entropies[2] < quadratic.compute_entropy() # verify entropy decrease


def test_clustering_basic(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    clustering = yoguen.Clustering(atoms)
    assert clustering.get_ncluster() == len(atoms)
    assert clustering.validate()


def test_clustering_compute_loss_identical(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']

    clustering = yoguen.Clustering(atoms)
    quadratic  = yoguen.Quadratic(atoms, hessian, geometry, cell)
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

    clustering = yoguen.Clustering(atoms)
    clustering.update_indices(indices)
    atoms_reduced = clustering.get_atoms_reduced()
    pos_reduced = atoms_reduced.get_positions()
    assert np.all(np.abs(pos_reduced - clusters) < 1e-4)
