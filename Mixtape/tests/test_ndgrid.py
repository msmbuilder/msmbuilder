import numpy as np
from mixtape.cluster import NDGrid

def test_ndgrid_1():
    X = np.array([-3, -2, -1, 1, 2, 3]).reshape(-1, 1)
    labels = NDGrid(n_bins_per_feature=2).fit([X]).predict([X])[0]
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))

def test_ndgrid_2():
    X = np.random.RandomState(0).randn(100, 2)
    labels = NDGrid(n_bins_per_feature=2, min=-5, max=5).fit([X]).predict([X])[0]
    
    mask0 = np.logical_and(X[:, 0] < 0, X[:, 1] < 0)
    assert np.all(labels[mask0] == 0)
    mask1 = np.logical_and(X[:, 0] > 0, X[:, 1] < 0)
    assert np.all(labels[mask1] == 1)
    mask2 = np.logical_and(X[:, 0] < 0, X[:, 1] > 0)
    assert np.all(labels[mask2] == 2)
    mask3 = np.logical_and(X[:, 0] > 0, X[:, 1] > 0)
    assert np.all(labels[mask3] == 3)


