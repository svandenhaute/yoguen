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
            (90, 180),
            (23, 130),
            (6, 450),
            ]
    clustering._score_pairs(quadratic, pairs)


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


def test_clustering_uio66(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    clustering = automap.Clustering(atoms)
    clustering.update_indices(indices) # set non-trivial mapping
    quadratic = automap.Quadratic(atoms, hessian, geometry, cell)
    entropies, quadratic_reduced = clustering.apply(quadratic)
    assert entropies[2] < quadratic.compute_entropy() # verify entropy decrease

    # create atoms_reduced and trivial clustering, merge pair
    #pair = (1, 5)
    #clustering_reduced = automap.Clustering(quadratic_reduced.atoms)
    #indices = list(clustering_reduced.get_indices())
    #indices[pair[0]] = indices[pair[0]] + indices.pop(pair[1]) # join 2 and 24
    #clustering_reduced.update_indices(tuple(indices))
    #entropies_, _ = clustering_reduced.apply(quadratic_reduced)
    #print(entropies_)

    #indices = list(clustering.get_indices())
    #indices[pair[0]] = indices[pair[0]] + indices.pop(pair[1]) # join 2 and 24
    #clustering.update_indices(tuple(indices))
    #entropies__, _ = clustering.apply(quadratic)
    #print(entropies__)


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


