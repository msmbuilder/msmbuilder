from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
from mixtape.cluster._kmedoids import kmedoids, triu_to_tril, contigify_ids, intertia_euclidean
from mixtape.cluster.kmedoids import _KMedoids, _MinibatchKMedoids

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


def test_2():
    random = np.random.RandomState(0)
    X = random.randn(10, 2)

    km = _KMedoids(n_clusters=2, n_passes=1).fit(X)

    intertia = 0
    for i in range(len(X)):
        intertia += euclidean(X[km.cluster_ids_[km.labels_[i]]], X[i])
    assert intertia == km.intertia_


def test_3():
    random = np.random.RandomState(0)
    X = random.randn(40, 2)
    X[20:] += 5

    k1 = _MinibatchKMedoids(n_clusters=2, max_iter=5, random_state=0, batch_size=20).fit(X)
    k2 = _KMedoids(n_clusters=2, n_passes=10).fit(X)

    assert (np.all(k1.labels_ == k2.labels_) or
            np.all(k1.labels_ == np.logical_not(k2.labels_)))


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
    ref = np.array([0, 1,  1,  2,  2,  3], dtype=np.int32)
    out, mapping = contigify_ids(inp)
    assert np.all(out == ref)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)
    assert mapping == {0:0, 10:1, 20:2, 21:3}


def test_contigify_ids_2():
    inp = np.array([2, 0, 10, 2, 2, 10], dtype=np.int32)
    ref = np.array([0, 1,  2, 0, 0,  2], dtype=np.int32)
    out, mapping = contigify_ids(inp)
    assert np.all(out == ref)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)
    assert mapping == {2:0, 0:1, 10:2}


def test_intertia_euclid():
    n = 20
    n_clusters = 5
    random = np.random.RandomState(0)
    X = random.rand(20, 2)
    cluster_ids = random.random_integers(
        low=0, high=n-1, size=n_clusters).astype(np.int32)
    labels = random.random_integers(
        low=0, high=n_clusters-1, size=n).astype(np.int32)

    ref = sum(euclidean(X[i], X[cluster_ids[labels[i]]]) for i in range(n))
    val = intertia_euclidean(X, cluster_ids, labels)
    assert ref == val