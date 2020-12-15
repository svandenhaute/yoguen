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
    pairs = []
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 1))
    pairs.append(yoguen.Pair.get_pair(clustering, 234, 13))
    pairs.append(yoguen.Pair.get_pair(clustering, 6, 450))
    pairs.append(yoguen.Pair.get_pair(clustering, 2, 3))
    pairlist = yoguen.PairList(pairs)
    scores = clustering.score_pairlist(
            pairlist,
            quadratic,
            progress=False,
            temperature=300,
            )

    # do manual calculation using apply()
    scores_manual = np.zeros(pairlist.npairs)
    indices = clustering.indices # keep track of original indices
    for i, pair in enumerate(pairlist):
        indices_ = pair.apply(indices)
        clustering.update_indices(indices_)
        entropies, _ = clustering.apply(quadratic)
        scores_manual[i] = entropies[1]
        clustering.update_indices(indices) # revert back to default clustering
    assert np.allclose(scores, scores_manual)


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


#def test_get_atoms_reduced(tmp_path):
#    system   = get_system('uio66') # get system input
#    atoms    = system['atoms']
#    geometry = system['geometry']
#    cell     = system['cell']
#    hessian  = system['hessian']
#    indices  = system['indices']
#    clusters = system['clusters']
#    #atoms.set_positions(geometry) # set atoms to minimum energy configuration
#    #atoms.set_cell(cell)
#
#    clustering = yoguen.Clustering(atoms)
#    clustering.update_indices(indices)
#    atoms_reduced = clustering.get_atoms_reduced()
#    pos_reduced = atoms_reduced.get_positions()
#    assert np.all(np.abs(pos_reduced - clusters) < 1e-4)
