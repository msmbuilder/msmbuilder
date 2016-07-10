from __future__ import print_function

import mdtraj as md
import numpy as np
import scipy.spatial.distance

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.libdistance import assign_nearest, cdist, pdist, dist, sumdist

random = np.random.RandomState()
VECTOR_METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev",
                  "canberra", "braycurtis", "hamming", "jaccard", "cityblock")


def setup():
    global X_double, Y_double, X_float, Y_float, X_rmsd, Y_rmsd, X_indices
    X_double = random.randn(10, 2)
    Y_double = random.randn(3, 2)
    X_float = random.randn(10, 2).astype(np.float32)
    Y_float = random.randn(3, 2).astype(np.float32)
    X_rmsd = AlanineDipeptide().get_cached().trajectories[0][0:10]
    Y_rmsd = AlanineDipeptide().get_cached().trajectories[0][30:33]
    X_rmsd.center_coordinates()
    Y_rmsd.center_coordinates()
    X_indices = random.randint(0, 10, size=5).astype(np.intp)


def test_assign_nearest_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            if metric == 'canberra' and X.dtype == np.float32:
                # this is tested separately
                continue

            assignments, inertia = assign_nearest(X, Y, metric)
            assert isinstance(assignments, np.ndarray)
            assert isinstance(inertia, float)

            cdist_1 = cdist(X, Y, metric=metric)
            assert cdist_1.shape == (10, 3)
            f = lambda: np.testing.assert_array_equal(
                    assignments,
                    cdist_1.argmin(axis=1))
            f.description = 'assign_nearest: %s %s' % (metric, X.dtype)

            yield lambda: np.testing.assert_almost_equal(
                    inertia,
                    cdist_1[np.arange(10), assignments].sum(),
                    decimal=5 if X.dtype == np.float32 else 10)


def test_assign_nearest_float_double_2():
    # test with X_indices

    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            if metric == 'canberra' and X.dtype == np.float32:
                # this is tested separately
                continue

            assignments, inertia = assign_nearest(X, Y, metric, X_indices)
            assert isinstance(assignments, np.ndarray)
            assert isinstance(inertia, float)

            cdist_1 = cdist(X[X_indices], Y, metric=metric)
            yield lambda: np.testing.assert_array_equal(
                    assignments,
                    cdist_1.argmin(axis=1))
            yield lambda: np.testing.assert_almost_equal(
                    inertia,
                    cdist_1[np.arange(5), assignments].sum(),
                    decimal=5 if X.dtype == np.float32 else 10)


def test_assign_nearest_rmsd_1():
    # rmsd assign nearest without X_indices
    assignments, inertia = assign_nearest(X_rmsd, Y_rmsd, "rmsd")
    assert isinstance(assignments, np.ndarray)
    assert isinstance(inertia, float)

    cdist_rmsd = cdist(X_rmsd, Y_rmsd, 'rmsd')
    assert cdist_rmsd.shape == (10, 3)

    np.testing.assert_array_equal(
            assignments,
            cdist_rmsd.argmin(axis=1))

    np.testing.assert_almost_equal(
            inertia,
            cdist_rmsd[np.arange(10), assignments].sum(),
            decimal=6)


def test_assign_nearest_rmsd_2():
    # rmsd assign nearest with X_indices
    assignments, inertia = assign_nearest(X_rmsd, Y_rmsd, "rmsd", X_indices)
    assert isinstance(assignments, np.ndarray)
    assert isinstance(inertia, float)

    cdist_rmsd = cdist(X_rmsd, Y_rmsd, 'rmsd')
    cdist_rmsd = cdist_rmsd[X_indices].astype(np.double)
    assert cdist_rmsd.shape == (5, 3)

    np.testing.assert_array_equal(
            assignments,
            cdist_rmsd.argmin(axis=1))

    np.testing.assert_almost_equal(
            inertia,
            cdist_rmsd[np.arange(5), assignments].sum(),
            decimal=5)


def test_cdist_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            cdist_1 = cdist(X, Y, metric)
            cdist_2 = scipy.spatial.distance.cdist(X, Y, metric)
            yield lambda: np.testing.assert_almost_equal(
                    cdist_1,
                    cdist_2,
                    decimal=5 if X.dtype == np.float32 else 10)


def test_pdist_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            pdist_1 = pdist(X, metric)
            pdist_2 = scipy.spatial.distance.pdist(X, metric)
            yield lambda: np.testing.assert_almost_equal(
                    pdist_1,
                    pdist_2,
                    decimal=5 if X.dtype == np.float32 else 10)


def test_pdist_double_float_2():
    # test with X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            pdist_1 = pdist(X, metric, X_indices=X_indices)
            pdist_2 = scipy.spatial.distance.pdist(X[X_indices], metric)
            yield lambda: np.testing.assert_almost_equal(
                    pdist_1,
                    pdist_2,
                    decimal=5 if X.dtype == np.float32 else 10)


def test_cdist_rmsd_1():
    got = cdist(X_rmsd, Y_rmsd, "rmsd")
    all2all = np.array([md.rmsd(X_rmsd, Y_rmsd[i], precentered=True)
                        for i in range(len(Y_rmsd))]).T
    np.testing.assert_almost_equal(got, all2all, decimal=5)


def test_pdist_rmsd_1():
    got = pdist(X_rmsd, "rmsd")
    all2all = np.array([md.rmsd(X_rmsd, X_rmsd[i], precentered=True)
                        for i in range(len(X_rmsd))])
    ref = all2all[np.triu_indices(10, k=1)]
    np.testing.assert_almost_equal(got, ref, decimal=5)


def test_pdist_rmsd_2():
    got = pdist(X_rmsd, "rmsd", X_indices)
    all2all = np.array([md.rmsd(X_rmsd, X_rmsd[i], precentered=True)
                        for i in range(len(X_rmsd))]).astype(np.double)
    submatrix = all2all[np.ix_(X_indices, X_indices)]

    ref = submatrix[np.triu_indices(5, k=1)]
    np.testing.assert_almost_equal(got, ref, decimal=4)


def test_dist_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            dist_1 = dist(X, Y[0], metric)
            dist_2 = cdist(X, Y, metric)[:, 0]
            yield lambda: np.testing.assert_almost_equal(
                    dist_1, dist_2,
                    decimal=5 if X.dtype == np.float32 else 10)


def test_dist_double_float_2():
    # test with X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            dist_1 = dist(X, Y[0], metric, X_indices)
            dist_2 = cdist(X, Y, metric)[X_indices, 0]
            yield lambda: np.testing.assert_almost_equal(
                    dist_1,
                    dist_2,
                    decimal=5 if X.dtype == np.float32 else 10)


def test_dist_rmsd_1():
    d = dist(X_rmsd, Y_rmsd[0], "rmsd")
    ref = md.rmsd(X_rmsd, Y_rmsd[0], precentered=True).astype(np.double)
    np.testing.assert_array_almost_equal(d, ref)


def test_dist_rmsd_2():
    d = dist(X_rmsd, Y_rmsd[0], "rmsd", X_indices)
    ref = (md.rmsd(X_rmsd, Y_rmsd[0], precentered=True)
           .astype(np.double)[X_indices])
    np.testing.assert_array_almost_equal(d, ref)


def test_sumdist_double_float():
    pairs = random.randint(0, 10, size=(5, 2)).astype(np.intp)
    for metric in VECTOR_METRICS:
        for X in (X_double, X_float):
            alldist = scipy.spatial.distance.squareform(pdist(X, metric))
            np.testing.assert_almost_equal(
                    sum(alldist[p[0], p[1]] for p in pairs),
                    sumdist(X, metric, pairs))


def test_sumdist_rmsd():
    pairs = random.randint(0, 10, size=(5, 2)).astype(np.intp)
    alldist = scipy.spatial.distance.squareform(pdist(X_rmsd, "rmsd"))
    np.testing.assert_almost_equal(
            sum(alldist[p[0], p[1]] for p in pairs),
            sumdist(X_rmsd, "rmsd", pairs),
            decimal=6)


def test_canberra_32_1():
    # with canberra in float32, there is a rounding issue where many of
    # the distances come out exactly the same, but due to finite floating
    # point resolution, a different one gets picked than by argmin()
    # on the cdist
    for i in range(10):
        X = random.randn(10, 2).astype(np.float32)
        Y = X[[0, 1, 2], :]

        assignments, inertia = assign_nearest(X, Y, 'canberra')
        cdist_can = cdist(X, Y, metric='canberra')
        ref = cdist_can.argmin(axis=1)
        if not np.all(ref == assignments):
            different = np.where(assignments != ref)[0]
            row = cdist_can[different, :]

            # if there are differences between assignments and the 'reference',
            # make sure that there is actually some difference between the
            # entries in that row of the distance matrix before throwing
            # an error
            if not np.all(row == row[0]):
                assert False


def test_canberra_32_2():
    for i in range(10):
        X = random.randn(10, 2).astype(np.float32)
        Y = X[[0, 1, 2], :]
        X_indices = (random.randint(0, 10, size=5)
                     .astype(np.intp))

        assignments, inertia = assign_nearest(X, Y, 'canberra',
                                              X_indices=X_indices)
        cdist_can = cdist(X[X_indices], Y, metric='canberra')
        ref = cdist_can.argmin(axis=1)
        if not np.all(ref == assignments):
            different = np.where(assignments != ref)[0]
            row = cdist_can[different, :]

            # if there are differences between assignments and the 'reference',
            # make sure that there is actually some difference between the
            # entries in that row of the distance matrix before throwing
            # an error
            if not np.all(row == row[0]):
                assert False
