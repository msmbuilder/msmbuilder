from __future__ import print_function, division, absolute_import

import numpy as np

from msmbuilder.utils import KDTree

X1 = 0.3 * np.random.RandomState(0).randn(500, 10)
X2 = 0.3 * np.random.RandomState(1).randn(1000, 10) + 10


def test_kdtree_k1():
    kdtree = KDTree([X1, X2])
    dists, inds = kdtree.query([
        [0] * 10,
        [10] * 10,
        [0] * 10
    ])

    assert len(inds) == 3
    for subind in inds:
        assert len(subind) == 2

    # traj i
    assert inds[0][0] == 0
    assert inds[1][0] == 1
    assert inds[2][0] == 0

    # framei
    assert 0 <= inds[0][1] < 500
    assert 0 <= inds[1][1] < 1000
    assert 0 <= inds[2][1] < 500

    # distances
    assert len(dists) == 3
    for d in dists:
        assert 0 <= d < 0.5


def test_kdtree_k2():
    kdtree = KDTree([X1, X2])
    dists, inds = kdtree.query([
        [0] * 10,
        [10] * 10,
        [0] * 10
    ], k=2)

    assert len(inds) == 3

    # traj i
    for qp in inds[0]: assert qp[0] == 0
    for qp in inds[1]: assert qp[0] == 1
    for qp in inds[2]: assert qp[0] == 0

    # frame i
    for qp in inds[0]: assert 0 <= qp[1] < 500
    for qp in inds[1]: assert 0 <= qp[1] < 1000
    for qp in inds[2]: assert 0 <= qp[1] < 500

    # distances
    assert len(dists) == 3
    for d in dists:
        assert 0 <= d[0] < 0.5
        assert 0 <= d[1] < 0.5
