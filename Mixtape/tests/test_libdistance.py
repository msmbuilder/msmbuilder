import numpy as np
import mdtraj as md
import scipy.spatial.distance
from mixtape.libdistance import assign_nearest, pdist, dist
from mixtape.datasets import AlanineDipeptide

random = np.random.RandomState(0)
X_double = random.randn(10, 2)
Y_double = random.randn(3, 2)
X_float = random.randn(10, 2).astype(np.float32)
Y_float = np.random.randn(3, 2).astype(np.float32)
X_rmsd = AlanineDipeptide().get().trajectories[0][0:10]
Y_rmsd = AlanineDipeptide().get().trajectories[0][10:13]

X_indices = random.random_integers(low=0, high=9, size=5)

VECTOR_METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev",
                  "canberra", "braycurtis", "hamming", "jaccard", "cityblock")


def test_assign_nearest_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            assignments, inertia = assign_nearest(X, Y, metric)
            assert isinstance(assignments, np.ndarray)
            assert isinstance(inertia, float)

            cdist = scipy.spatial.distance.cdist(X, Y, metric=metric)
            assert cdist.shape == (10, 3)
            yield lambda: np.testing.assert_array_equal(
                assignments,
                cdist.argmin(axis=1))

            yield lambda: np.testing.assert_almost_equal(
                inertia,
                cdist[np.arange(10), assignments].sum(),
                decimal=5)


def test_assign_nearest_float_double_2():
    # test with X_indices

    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):

            assignments, inertia = assign_nearest(X, Y, metric, X_indices)
            assert isinstance(assignments, np.ndarray)
            assert isinstance(inertia, float)

            cdist = scipy.spatial.distance.cdist(X[X_indices], Y, metric=metric)
            yield lambda: np.testing.assert_array_equal(
                assignments,
                cdist.argmin(axis=1))
            yield lambda: np.testing.assert_almost_equal(
                inertia,
                cdist[np.arange(5), assignments].sum(),
                decimal=5)


def test_assign_nearest_rmsd_1():
    # rmsd assign nearest without X_indices
    assignments, inertia = assign_nearest(X_rmsd, Y_rmsd, "rmsd")
    assert isinstance(assignments, np.ndarray)
    assert isinstance(inertia, float)

    cdist = np.array([md.rmsd(X_rmsd, Y_rmsd[i]) for i in range(len(Y_rmsd))]).T
    assert cdist.shape == (10, 3)

    np.testing.assert_array_equal(
        assignments,
        cdist.argmin(axis=1))

    np.testing.assert_almost_equal(
        inertia,
        cdist[np.arange(10), assignments].sum(),
        decimal=5)


def test_assign_nearest_rmsd_2():
    # rmsd assign nearest with X_indices
    assignments, inertia = assign_nearest(X_rmsd, Y_rmsd, "rmsd", X_indices)
    assert isinstance(assignments, np.ndarray)
    assert isinstance(inertia, float)

    cdist = np.array([md.rmsd(X_rmsd, Y_rmsd[i]) for i in range(len(Y_rmsd))]).T
    cdist = cdist[X_indices]
    assert cdist.shape == (5, 3)

    np.testing.assert_array_equal(
        assignments,
        cdist.argmin(axis=1))

    np.testing.assert_almost_equal(
        inertia,
        cdist[np.arange(5), assignments].sum(),
        decimal=5)


def test_pdist_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            pdist_1 = pdist(X, metric)
            pdist_2 = scipy.spatial.distance.pdist(X, metric)
            yield lambda : np.testing.assert_almost_equal(
                pdist_1,
                pdist_2,
                decimal=5)


def test_pdist_double_float_2():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            pdist_1 = pdist(X, metric, X_indices=X_indices)
            pdist_2 = scipy.spatial.distance.pdist(X[X_indices], metric)
            yield lambda : np.testing.assert_almost_equal(
                pdist_1,
                pdist_2,
                decimal=5)


def test_pdist_rmsd_1():
    got = pdist(X_rmsd, "rmsd")
    all2all = np.array([md.rmsd(X_rmsd, X_rmsd[i]) for i in range(len(X_rmsd))])
    ref = all2all[np.triu_indices(10, k=1)]
    np.testing.assert_almost_equal(got, ref, decimal=5)


def test_pdist_rmsd_2():
    got = pdist(X_rmsd, "rmsd", X_indices)
    all2all = np.array([md.rmsd(X_rmsd, X_rmsd[i])
                        for i in range(len(X_rmsd))]).astype(np.double)
    submatrix = all2all[np.ix_(X_indices, X_indices)]

    ref = submatrix[np.triu_indices(5, k=1)]
    np.testing.assert_almost_equal(got, ref, decimal=4)


def test_dist_double_float_1():
    # test without X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            dist_1 = dist(X, Y[0], metric)
            dist_2 = scipy.spatial.distance.cdist(X, [Y[0]], metric)[:,0]
            yield lambda : np.testing.assert_almost_equal(
                dist_1,
                dist_2,
                decimal=5)


def test_dist_double_float_2():
    # test with X_indices
    for metric in VECTOR_METRICS:
        for X, Y in ((X_double, Y_double), (X_float, Y_float)):
            dist_1 = dist(X, Y[0], metric, X_indices)
            dist_2 = scipy.spatial.distance.cdist(X, [Y[0]], metric)[X_indices,0]
            yield lambda : np.testing.assert_almost_equal(
                dist_1,
                dist_2,
                decimal=5)


def test_dist_rmsd_1():
    d = dist(X_rmsd, Y_rmsd[0], "rmsd")
    ref = md.rmsd(X_rmsd, Y_rmsd[0]).astype(np.double)
    np.testing.assert_array_almost_equal(d, ref)


def test_dist_rmsd_2():
    d = dist(X_rmsd, Y_rmsd[0], "rmsd", X_indices)
    ref = md.rmsd(X_rmsd, Y_rmsd[0]).astype(np.double)[X_indices]
    np.testing.assert_array_almost_equal(d, ref)
