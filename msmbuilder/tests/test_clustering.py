from __future__ import print_function

import mdtraj as md
import mdtraj.testing
import numpy as np
import scipy.spatial.distance

import msmbuilder.cluster

X1 = 0.3 * np.random.RandomState(0).randn(1000, 10).astype(np.double)
X2 = 0.3 * np.random.RandomState(1).randn(1000, 10).astype(np.float32)
trj = md.load(md.testing.get_fn("traj.h5"))


def test_regular_spatial_rmsd():
    model = msmbuilder.cluster.RegularSpatial(d_min=0.01, metric='rmsd')
    model.fit([trj])

    assert isinstance(model.cluster_centers_, md.Trajectory)
    assert len(model.cluster_centers_) == model.n_clusters_
    predict = model.predict([trj])
    assert isinstance(predict, list) and len(predict) == 1
    assert len(predict[0]) == len(trj)
    assert isinstance(predict[0], np.ndarray) and predict[0].dtype == np.intp


def test_regular_spatial():
    model = msmbuilder.cluster.RegularSpatial(d_min=0.8)

    for X in [X1, X2]:
        model.fit([X])

        assert model.cluster_centers_.shape[1] == 10
        assert isinstance(model.cluster_centers_, np.ndarray)
        assert len(model.cluster_centers_) == model.n_clusters_
        predict = model.predict([X])
        assert isinstance(predict, list) and len(predict) == 1
        assert len(predict[0]) == len(X)
        assert (isinstance(predict[0], np.ndarray)
                and predict[0].dtype == np.intp)

        assert model.cluster_centers_.shape[0] > 200
        assert not np.all(scipy.spatial.distance.pdist(X) > model.d_min)
        assert np.all(scipy.spatial.distance.pdist(model.cluster_centers_)
                      > model.d_min)

        assert np.all(np.shape(model.cluster_center_indices_)
                      == (len(model.cluster_center_indices_), 2))


def test_kcenters_rmsd():
    model = msmbuilder.cluster.KCenters(3, metric='rmsd')
    model.fit([trj])

    assert len(model.cluster_centers_) == 3
    assert isinstance(model.cluster_centers_, md.Trajectory)
    predict = model.predict([trj])
    assert isinstance(predict, list) and len(predict) == 1
    assert len(predict[0]) == len(trj)
    assert isinstance(predict[0], np.ndarray) and predict[0].dtype == np.intp


def test_kcenters_spatial():
    model = msmbuilder.cluster.KCenters(5)

    for X in [X1, X2]:
        model.fit([X])

        assert model.cluster_centers_.shape[1] == 10
        assert isinstance(model.cluster_centers_, np.ndarray)
        assert len(model.cluster_centers_) == 5
        predict = model.predict([X])
        assert isinstance(predict, list) and len(predict) == 1
        assert len(predict[0]) == len(X)
        assert (isinstance(predict[0], np.ndarray)
                and predict[0].dtype == np.intp)
