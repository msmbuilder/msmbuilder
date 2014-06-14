import numpy as np
from scipy.spatial.distance import pdist, squareform
from mdtraj.utils import timing
from mixtape.cluster import RegularSpatial

X = 0.3*np.random.RandomState(0).randn(1000, 10).astype(np.float64)
Y = 0.3*np.random.RandomState(0).randn(1000, 10).astype(np.float32)

def test_1():
    x = np.arange(10).reshape(10,1)

    model = RegularSpatial(d_min=0.99)
    model.fit([x])

    assert len(model.cluster_centers_) == 10
    np.testing.assert_array_equal(x, model.cluster_centers_)                                  


def test_2():
    # test that the optimized version actually gives the same results
    c1 = RegularSpatial(d_min=1.0, opt=True).fit([X]).cluster_centers_
    c2 = RegularSpatial(d_min=1.0, opt=False).fit([X]).cluster_centers_
    c3 = RegularSpatial(d_min=1.0, opt=True).fit([Y]).cluster_centers_
    c4 = RegularSpatial(d_min=1.0, opt=False).fit([Y]).cluster_centers_
    np.testing.assert_array_almost_equal(c1, c2)
    np.testing.assert_array_almost_equal(c1, c3)
    np.testing.assert_array_almost_equal(c1, c4)


def test_3():
    # test that all the centers are farther than d_min from each other

    for x in [X, Y]:
        for d_min in [0.9, 1, 1.1]:
            c1 = RegularSpatial(d_min=d_min, opt=True).fit([x]).cluster_centers_
            D = squareform(pdist(c1))

            # the only distances less than d_min in the all to all distance
            # matrix of the cluster centers should be the diagonal elements of the
            # matrix
            ix, jx = np.where(D < d_min)
            refix, refjx = np.diag_indices_from(D)
            np.testing.assert_array_equal(ix, refix)
            np.testing.assert_array_equal(jx, refjx)


def test_4():
    # test that the two code paths in predict() give the same result
    for x in [X, Y]:
        model = RegularSpatial(d_min=0.8, opt=True)
        model.fit([x])
        l1 = model.predict([x])

        model.opt = False
        l2 = model.predict([x])
        np.testing.assert_array_equal(l1, l2)
    
    
