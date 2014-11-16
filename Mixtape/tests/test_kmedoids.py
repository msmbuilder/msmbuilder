from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
from mixtape.cluster._kmedoids import kmedoids, triu_to_tril, contigify_ids


def test_1():
    random = np.random.RandomState(0)

    n_elements = 10
    n_clusters = 2
    n_pass = 5
    X = random.randn(n_elements, 2)
    distmatrix = pdist(X)
    clusterid = random.randint(2, size=10).astype(np.int32)

    cid, i1, f = kmedoids(n_clusters, distmatrix, n_pass)
    cid, i2, f = kmedoids(n_clusters, distmatrix, n_pass, cid)
    assert np.all(np.unique(cid) == np.arange(n_clusters))


def test_triu_to_tril_1():
    n = 10
    random = np.random.RandomState(0)
    X = random.randn(n, 1)
    d = pdist(X)
    lower = triu_to_tril(d)

    triu_indices = np.triu_indices(n, 1)
    assert np.all(squareform(d)[triu_indices] == d)

    tril_indices = np.tril_indices(n, -1)
    assert np.all(squareform(d)[tril_indices] == lower)


def test_contigify_ids_1():
    inp = np.array([0, 10, 10, 20, 20, 21], dtype=np.int32)
    out = np.array([0, 1,  1,  2,  2,  3], dtype=np.int32)
    assert np.all(contigify_ids(inp) == out)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)


def test_contigify_ids_2():
    inp = np.array([2, 0, 10, 2, 2, 10], dtype=np.int32)
    out = np.array([0, 1,  2, 0, 0,  2], dtype=np.int32)
    assert np.all(contigify_ids(inp) == out)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)
