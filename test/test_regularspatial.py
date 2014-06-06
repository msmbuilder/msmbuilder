import numpy as np
from scipy.spatial.distance import pdist, squareform
from mdtraj.utils import timing
from mixtape.cluster import RegularSpatial

X = 0.3*np.random.RandomState(0).randn(10000, 10)

def test_1():
    x = np.arange(10).reshape(10,1)

    model = RegularSpatial(d_min=0.99)
    model.fit([x])

    assert len(model.cluster_centers_) == 10
    np.testing.assert_array_equal(x, model.cluster_centers_)                                  


def test_2():
    # test that the optimized version actually gives the same results

    with timing('opt=True'):
        c1 = RegularSpatial(d_min=1.0, opt=True).fit([X]).cluster_centers_
    with timing('opt=False'):
        c2 = RegularSpatial(d_min=1.0, opt=False).fit([X]).cluster_centers_

    np.testing.assert_array_equal(c1, c2)
    print c1.shape


def test_3():
    # test that all the centers are farther than d_min from each other

    for d_min in [0.9, 1, 1.1]:
        c1 = RegularSpatial(d_min=d_min, opt=True).fit([X]).cluster_centers_
        D = squareform(pdist(c1))

        # the only distances less than d_min in the all to all distance
        # matrix of the cluster centers should be the diagonal elements of the
        # matrix
        ix, jx = np.where(D < d_min)
        refix, refjx = np.diag_indices_from(D)
        np.testing.assert_array_equal(ix, refix)
        np.testing.assert_array_equal(jx, refjx)

