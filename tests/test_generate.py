import yoguen
import numpy as np

from systems import get_system


def test_generate_select(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']
    geometry = system['geometry']
    cell     = system['cell']
    atoms.set_positions(geometry)
    atoms.set_cell(cell)

    clustering = yoguen.Clustering(atoms)
    generator  = yoguen.PairGenerator(
            cutoff=1.0,
            max_neighbors=1,
            tol_distance=1e-3,
            )
    clist  = generator.compute_candidates(clustering) # includes only O-H bonds
    scores = np.linspace(1.1, 1, len(clist)) # identical scores > 0, not sorted
    assert len(clist) % 2 == 0 # should find even number of OH bonds
    n = len(clist) // 2
    scores[n:] = 2.0 # disqualify second half
    selection, _within_tol = generator.select(
            clist,
            scores,
            tol_score=5e-1, # should include half of candidates
            )
    assert len(selection) == n
    assert len(_within_tol) == 0 # all pairs are equiv


def test_pair_init():
    # create random pair
    i = 234
    j = 12
    elements = (118, 117)
    groups = (
            set([1, 2]),
            set([12, 11, 3]),
            )
    identities = list(zip(elements, groups))
    distance = 5.0 # is required to be float
    pair = yoguen.Pair(
            i,
            j,
            identities,
            distance,
            )
    pair_ = yoguen.Pair(
            j, # first index larger than second
            i,
            identities[::-1],
            distance,
            )
    assert pair[0] == pair_[0]
    assert pair[1] == pair_[1]
    assert pair.identities[0] == pair_.identities[0]
    assert pair.identities[1] == pair_.identities[1]


def test_get_pair():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pair = yoguen.Pair.get_pair(clustering, 35, 300)
    assert pair[0] < pair[1]


def test_pair_apply_repeat():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pair0 = yoguen.Pair.get_pair(clustering, 35, 300)
    pair1= yoguen.Pair.get_pair(clustering, 30, 29) # unaffected by first pair
    indices = pair0.apply(clustering.indices)
    indices = pair1.apply(indices)
    clustering.update_indices(indices)
    assert len(clustering.indices) == len(atoms) - 2
    assert clustering.indices[29] == tuple([29, 30])
    assert clustering.indices[29] == tuple([29, 30])
    assert clustering.indices[400] == tuple([402])
    assert clustering.indices[34] == tuple([35, 300])


def test_pair_clist_apply():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pair0 = yoguen.Pair.get_pair(clustering, 35, 300)
    pair1 = yoguen.Pair.get_pair(clustering, 0, 1)
    pair2 = yoguen.Pair.get_pair(clustering, 18, 30)
    pair3 = yoguen.Pair.get_pair(clustering, 455, 15)
    clist = [pair0, pair1, pair2, pair3]

    indices = yoguen.Pair.apply_clist(clist, clustering.indices)
    clustering.update_indices(indices) # perform validation
    assert len(indices) == len(atoms) - len(clist)
    assert indices[0]  == tuple([0, 1])
    assert indices[14] == tuple([15, 455])
    assert indices[17] == tuple([18, 30])
    assert indices[33] == tuple([35, 300])


def test_pair_equivalent():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pair0 = yoguen.Pair.get_pair(clustering, 35, 300)
    pair1 = yoguen.Pair.get_pair(clustering, 0, 1)
    assert not yoguen.Pair.possibly_equivalent(
            pair0,
            pair1,
            )
    pair0 = yoguen.Pair.get_pair(clustering, 0, 1)
    pair1 = yoguen.Pair.get_pair(clustering, 0, 1)
    assert yoguen.Pair.possibly_equivalent(
            pair0,
            pair1,
            )
    pair2 = yoguen.Pair.get_pair(clustering, 0, 450)
    assert not yoguen.Pair.possibly_equivalent(
            pair0,
            pair2,
            )
    pair1.distance = pair0.distance * 2
    assert not yoguen.Pair.possibly_equivalent(
            pair0,
            pair1,
            )
    assert yoguen.Pair.possibly_equivalent(
            pair0,
            pair1,
            tol_distance=10,
            )


def test_generate_pairs_uio66(tmp_path):
    system   = get_system('uio66')
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    generator  = yoguen.PairGenerator(3, 100)
    clist = generator.compute_candidates(clustering)

    # ensure pairs are sorted
    for pair in clist:
        assert pair[0] < pair[1]

    # ensure pairs are unique
    index = 0
    while index < len(clist) - 1:
        pair = clist[index]
        for i in range(index + 1, len(clist)):
            pair_ = clist[i]
            assert (pair[0] != pair_[0]) or (pair[1] != pair_[1])
        index += 1
