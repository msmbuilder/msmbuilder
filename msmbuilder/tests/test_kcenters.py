from __future__ import division

import numpy as np
import scipy.spatial.distance
from mdtraj.testing import eq

from msmbuilder import libdistance
from msmbuilder.cluster import KCenters


def test_shapes():
    # make sure all the shapes are correct of the fit parameters
    m = KCenters(n_clusters=3)
    m.fit([np.random.randn(23, 2), np.random.randn(10, 2)])

    assert isinstance(m.labels_, list)
    assert isinstance(m.distances_, list)
    assert len(m.labels_) == 2
    eq(m.cluster_centers_.shape, (3, 2))
    eq(m.labels_[0].shape, (23,))
    eq(m.labels_[1].shape, (10,))
    eq(m.distances_[0].shape, (23,))
    eq(m.distances_[1].shape, (10,))

    eq(m.fit_predict([np.random.randn(10, 2)])[0].shape, (10,))
    assert np.all(np.logical_not(np.isnan(m.distances_[0])))


def test_three_clusters():
    # some data at (0,0), some data at (1,1) and some data at (0.5, 0.5)
    data = [np.zeros((10, 2)), np.ones((10, 2)), 0.5 * np.ones((10, 2))]

    m = KCenters(n_clusters=2, random_state=0)
    m.fit(data)

    # the centers should be [0,0], [1,1] (in either order). This
    # assumes that the random state seeded the initial center at
    # either (0,0) or (1,1). A different random state could have
    # seeded the first cluster at [0.5, 0.5]
    assert np.all(m.cluster_centers_ == np.array([[0, 0], [1, 1]])) or \
           np.all(m.cluster_centers_ == np.array([[1, 1], [0, 0]]))

    # the distances should be 0 or sqrt(2)/2
    eq(np.unique(np.concatenate(m.distances_)), np.array([0, np.sqrt(2) / 2]))


def test_euclidean():
    # test for predict using euclidean distance

    m = KCenters(n_clusters=10)
    data = np.random.randn(100, 2)
    labels1 = m.fit_predict([data])
    labels2 = m.predict([data])

    eq(labels1[0], labels2[0])
    all_pairs = scipy.spatial.distance.cdist(data, m.cluster_centers_)
    eq(labels2[0], np.argmin(all_pairs, axis=1))


def test_cityblock():
    # test for predict() using non-euclidean distance. because of the
    # way the code is structructured, this takes a different path
    model = KCenters(n_clusters=10, metric='cityblock')
    data = np.random.randn(100, 2)
    labels1 = model.fit_predict([data])
    labels2 = model.predict([data])

    eq(labels1[0], labels2[0])
    all_pairs = scipy.spatial.distance.cdist(data, model.cluster_centers_,
                                             metric='cityblock')
    eq(labels2[0], np.argmin(all_pairs, axis=1))


def test_sqeuclidean():
    model1 = KCenters(n_clusters=10, random_state=0, metric='euclidean')
    model2 = KCenters(n_clusters=10, random_state=0, metric='sqeuclidean')

    data = np.random.RandomState(0).randn(100, 2)
    eq(model1.fit_predict([data])[0], model2.fit_predict([data])[0])


def test_fit_predict():
    # are fit_predict and fit().predict() consistent?
    trj = np.random.RandomState(0).randn(30, 2)
    k = KCenters(n_clusters=10, random_state=0).fit([trj])
    l1 = KCenters(n_clusters=10, random_state=0).fit([trj]).predict([trj])[0]
    l2 = KCenters(n_clusters=10, random_state=0).fit_predict([trj])[0]

    eq(l1, l2)


def test_dtype():
    X = np.random.RandomState(1).randn(100, 2)
    X32 = X.astype(np.float32)
    X64 = X.astype(np.float64)
    m1 = KCenters(n_clusters=10, random_state=0).fit([X32])
    m2 = KCenters(n_clusters=10, random_state=0).fit([X64])

    eq(m1.cluster_centers_, m2.cluster_centers_)
    eq(m1.distances_[0], m2.distances_[0])
    eq(m1.labels_[0], m2.labels_[0])
    assert np.all(np.logical_not(np.isnan(m1.distances_[0])))
    eq(m1.predict([X32])[0], m2.predict([X64])[0])
    eq(m1.predict([X32])[0], m1.labels_[0])
    eq(float(m1.inertia_),
       libdistance.assign_nearest(X32, m1.cluster_centers_, "euclidean")[1])
