import yoguen
import numpy as np

from systems import get_system


def test_pair_init():
    # create random pair
    i = 23
    j = 123
    types = (118, 117)
    atom_types = [ # should be list of tuples
            (1, 2, 1),
            (12, 11, 3),
            ]
    atom_indices = [ # should be list of tuples, without duplicates
            (399, 400, 401),
            (1, 2, 3),
            ]
    distance = 5.0 # is required to be float
    pair = yoguen.Pair(
            i,
            j,
            types,
            atom_indices,
            atom_types,
            distance,
            )
    pair_ = yoguen.Pair(
            j, # first index larger than second, should reverse internally
            i,
            types[::-1],
            atom_indices[::-1],
            atom_types[::-1],
            distance,
            )
    assert pair_._i == i
    assert pair[0] == pair_[0]
    assert pair[1] == pair_[1]
    assert pair.types[1] == pair_.types[1]
    assert pair.atom_indices[0] == pair_.atom_indices[0]
    assert pair.atom_types[0] == pair_.atom_types[0]


def test_get_pair():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    clustering = yoguen.Clustering(atoms)
    pair = yoguen.Pair.get_pair(clustering, 35, 300)
    assert pair[0] < pair[1]


def test_pairlist_apply():
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pair0 = yoguen.Pair.get_pair(clustering, 35, 300)
    pair1 = yoguen.Pair.get_pair(clustering, 0, 1)
    pair2 = yoguen.Pair.get_pair(clustering, 18, 30)
    pair3 = yoguen.Pair.get_pair(clustering, 455, 15)
    pairs = [pair0, pair1, pair2, pair3]

    pairlist = yoguen.PairList(pairs)
    indices = pairlist.apply(clustering.indices)
    clustering.update_indices(indices)
    assert len(indices) == len(atoms) - pairlist.npairs
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
    assert not yoguen.Pair.test_equivalence(
            pair0,
            pair1,
            )
    pair0 = yoguen.Pair.get_pair(clustering, 0, 1)
    pair1 = yoguen.Pair.get_pair(clustering, 0, 1)
    assert yoguen.Pair.test_equivalence(
            pair0,
            pair1,
            )
    pair2 = yoguen.Pair.get_pair(clustering, 0, 450)
    assert not yoguen.Pair.test_equivalence(
            pair0,
            pair2,
            )
    pair1.distance = pair0.distance * 2
    assert not yoguen.Pair.test_equivalence(
            pair0,
            pair1,
            )
    assert yoguen.Pair.test_equivalence(
            pair0,
            pair1,
            tol_distance=10,
            )


def test_generate_pairs_uio66(tmp_path):
    system = get_system('uio66')
    atoms  = system['atoms']

    clustering = yoguen.Clustering(atoms)
    cutoff = 3 # cutoff radius in angstrom
    max_neighbors = 1 # maximum number of neighbors to consider for each atom
    pairlist = yoguen.PairList.generate(clustering, cutoff, max_neighbors)

    # ensure pairs are sorted
    for i in range(pairlist.npairs):
        assert pairlist[i][0] < pairlist[i][1]

    # ensure pairs are unique
    index = 0
    while index < pairlist.npairs - 1:
        pair = pairlist[index]
        for i in range(index + 1, pairlist.npairs):
            pair_ = pairlist[i]
            assert (pair[0] != pair_[0]) or (pair[1] != pair_[1])
        index += 1


def test_pairlist_filter_disjunct(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pairs = []
    pairs.append(yoguen.Pair.get_pair(clustering, 35, 300))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 1))
    pairs.append(yoguen.Pair.get_pair(clustering, 1, 30))
    pairs.append(yoguen.Pair.get_pair(clustering, 455, 15))
    pairs.append(yoguen.Pair.get_pair(clustering, 455, 15))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 34))
    pairlist = yoguen.PairList(pairs)
    pairlist.filter_disjunct()
    assert pairlist.npairs == 3 # three duplicates
    assert pairlist._pairs[2]._i == 15
    assert pairlist._pairs[2]._j == 455


def test_pairlist_filter_score(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pairs = []
    pairs.append(yoguen.Pair.get_pair(clustering, 35, 300))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 1))
    pairs.append(yoguen.Pair.get_pair(clustering, 1, 30))
    pairs.append(yoguen.Pair.get_pair(clustering, 455, 15))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 34))
    pairlist = yoguen.PairList(pairs)
    scores = np.array([0.1, 0.2, 0.2, 0.3, 0.04])
    pairlist.add_scores(scores)
    pairlist.filter_scores(0.2)
    assert pairlist.npairs == 4 # score 0.2 should be included
    assert pairlist._pairs[0]._i == 0
    assert pairlist._pairs[0]._j == 34
    assert pairlist._pairs[3]._i == 1
    assert pairlist._pairs[3]._j == 30


def test_pairlist_filter_equivalent(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']

    clustering = yoguen.Clustering(atoms)
    pairs = []
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 1)) # Zr - Zr
    pairs.append(yoguen.Pair.get_pair(clustering, 2, 3)) # Zr - Zr; EQUIV
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 20)) # Zr - Zr; NONEQUIV
    pairs.append(yoguen.Pair.get_pair(clustering, 1, 450)) # Zr - H
    pairs.append(yoguen.Pair.get_pair(clustering, 455, 454)) # H - H
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 34)) # Zr - O
    pairlist = yoguen.PairList(pairs)
    pairlist.filter_equivalent(pairs[0])
    assert pairlist.npairs == 2
    assert pairlist._pairs[0]._i == 0
    assert pairlist._pairs[0]._j == 1
    assert pairlist._pairs[1]._i == 2
    assert pairlist._pairs[1]._j == 3


def test_pairlist_iterate(tmp_path):
    system   = get_system('uio66') # get system input
    atoms    = system['atoms']
    clustering = yoguen.Clustering(atoms)
    pairs = []
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 1))
    pairs.append(yoguen.Pair.get_pair(clustering, 2, 3))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 20))
    pairs.append(yoguen.Pair.get_pair(clustering, 1, 450))
    pairs.append(yoguen.Pair.get_pair(clustering, 455, 454))
    pairs.append(yoguen.Pair.get_pair(clustering, 0, 34))
    pairlist = yoguen.PairList(pairs)
    count = 0
    for pair in pairlist:
        count += 1
    for pair in pairlist:
        count += 1
    assert count == 2 * pairlist.npairs

