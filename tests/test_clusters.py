import automap
import numpy as np

from systems import get_system


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

    # compute loss for identical mapping --> should be zero
    smap = automap.Reduction.compute_loss(clustering, quadratic, T=300)
    assert smap == 0.0


def test_clustering_uio66(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']

    clustering = automap.Clustering(atoms)
    clustering.update_indices(indices) # set non-trivial mapping
    quadratic  = automap.Quadratic(atoms, hessian, geometry, cell)

    # compute loss for this mapping
    smap = automap.Reduction.compute_loss(clustering, quadratic, T=300)
    assert abs(smap - 3.946) < 1e-1


def test_get_atoms_reduced(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    hessian  = system['hessian']
    indices  = system['indices']
    clusters = system['clusters']

    atoms.set_positions(geometry)
    clustering = automap.Clustering(atoms)
    clustering.update_indices(indices)
    atoms_reduced = clustering.get_atoms_reduced()
    pos_reduced = atoms_reduced.get_positions()
    assert np.all(np.abs(pos_reduced - clusters) < 1e-4)
