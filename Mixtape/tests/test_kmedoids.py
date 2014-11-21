from __future__ import print_function, absolute_import, division

import numpy as np
from numpy.testing import assert_raises
from scipy.spatial.distance import pdist, squareform, euclidean
from mixtape.cluster._kmedoids import kmedoids, contigify_ids
from mixtape.cluster.kmedoids import _KMedoids
from mixtape.cluster.minibatchkmedoids import _MiniBatchKMedoids
from mixtape import libdistance


def test_inertia():
    random = np.random.RandomState(0)
    X = random.randn(10, 2)

    for km in (_KMedoids(n_clusters=3, n_passes=5),
               _MiniBatchKMedoids(n_clusters=3)):
        km.fit(X)
        inertia = 0
        for i in range(len(X)):
            inertia += euclidean(X[km.cluster_ids_[km.labels_[i]]], X[i])

        np.testing.assert_almost_equal(inertia, km.inertia_)


def test_obvious_clustering():
    random = np.random.RandomState(0)
    X = random.randn(40, 2)
    X[20:] += 5

    k1 = _MiniBatchKMedoids(n_clusters=2, max_iter=5, random_state=0, batch_size=20).fit(X)
    k2 = _KMedoids(n_clusters=2, n_passes=10).fit(X)

    assert (np.all(k1.labels_ == k2.labels_) or
            np.all(k1.labels_ == np.logical_not(k2.labels_)))


def test_4():
    assert_raises(ValueError, lambda: _MiniBatchKMedoids(metric='sdf').fit(np.zeros((10,2))))
    assert_raises(ValueError, lambda: _KMedoids(metric='sdf').fit(np.zeros((10,2))))


def test_contigify_ids_1():
    inp = np.array([0, 10, 10, 20, 20, 21])
    ref = np.array([0, 1,  1,  2,  2,  3])
    out, mapping = contigify_ids(inp)
    assert np.all(out == ref)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)
    assert mapping == {0:0, 10:1, 20:2, 21:3}


def test_contigify_ids_2():
    inp = np.array([2, 0, 10, 2, 2, 10])
    ref = np.array([0, 1,  2, 0, 0,  2])
    out, mapping = contigify_ids(inp)
    assert np.all(out == ref)
    # it's inplace, so they should be equal now
    assert np.all(inp == out)
    assert mapping == {2:0, 0:1, 10:2}


def test_index():
    n = 50
    def q(i, j):
        if i == j:
            raise ValueError()
        if (i < j):
            return int(n*i - i*(i+1)/2 + j - 1 - i)
        return int(n*j - j*(j+1)/2 + i - 1 - j)

    
    X = np.random.randn(n, 1)
    pdist = libdistance.pdist(X, 'euclidean')
    for i in range(n):
        for j in range(n):
            if (i != j):
                assert euclidean(X[i], X[j]) == pdist[q(i,j)]

