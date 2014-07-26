from __future__ import division
import os
import operator
import numpy as np
import mdtraj as md
from functools import reduce
from mdtraj.testing import eq
from sklearn.externals.joblib import load
from mixtape.cluster import KCenters
import scipy.spatial.distance

def test_kcenters_1():
    # make sure all the shapes are correct of the fit parameters


    for m in [KCenters(n_clusters=3, opt=True), KCenters(n_clusters=3, opt=False)]:
        m.fit([np.random.randn(23,2), np.random.randn(10,2)])

        assert isinstance(m.labels_, list)
        assert isinstance(m.distances_, list)
        assert len(m.labels_) == 2
        eq(m.cluster_centers_.shape, (3,2))
        eq(m.labels_[0].shape, (23,))
        eq(m.labels_[1].shape, (10,))
        eq(m.distances_[0].shape, (23,))
        eq(m.distances_[1].shape, (10,))

        eq(m.fit_predict([np.random.randn(10, 2)])[0].shape, (10,))
        assert np.all(np.logical_not(np.isnan(m.distances_[0])))


def test_kcenters_2():
    # some data at (0,0), some data at (1,1) and some data at (0.5, 0.5)
    data = [np.zeros((10,2)), np.ones((10,2)), 0.5*np.ones((10,2))]

    for opt in [True, False]:
        m = KCenters(n_clusters=2, random_state=0, opt=opt)
        m.fit(data)

        # the centers should be [0,0], [1,1] (in either order). This
        # assumes that the random state seeded the initial center at
        # either (0,0) or (1,1). A different random state could have
        # seeded the first cluster at [0.5, 0.5]
        assert np.all(m.cluster_centers_ == np.array([[0,0], [1,1]])) or \
            np.all(m.cluster_centers_ == np.array([[1,1], [0,0]]))

        # the distances should be 0 or sqrt(2)/2
        eq(np.unique(np.concatenate(m.distances_)), np.array([0, np.sqrt(2)/2]))


def test_kcenters_3():
    # test for predict using euclidean distance

    for model in [KCenters(n_clusters=10, opt=True), KCenters(n_clusters=10, opt=False)]:
        data = np.random.randn(100, 2)
        labels1 = model.fit_predict([data])
        labels2 = model.predict([data])

        eq(labels1[0], labels2[0])
        all_pairs = scipy.spatial.distance.cdist(data, model.cluster_centers_)
        eq(labels2[0], np.argmin(all_pairs, axis=1))


def test_kcenters_4():
    # test for predict() using non-euclidean distance. because of the
    # way the code is structructured, this takes a different path
    model = KCenters(n_clusters=10, metric='cityblock')
    data = np.random.randn(100, 2)
    labels1 = model.fit_predict([data])
    labels2 = model.predict([data])

    eq(labels1[0], labels2[0])
    all_pairs = scipy.spatial.distance.cdist(data, model.cluster_centers_, metric='cityblock')
    eq(labels2[0], np.argmin(all_pairs, axis=1))


def test_kcenters_5():
    # test custom metric. this is a euclidean metric vs. a squared euclidean metric (should give)
    # the same assignments
    model1 = KCenters(n_clusters=10, random_state=0, metric='euclidean')
    model2 = KCenters(n_clusters=10, random_state=0, metric=lambda target, ref, i: np.sum((target-ref[i])**2, axis=1))

    data = np.random.RandomState(0).randn(100, 2)
    eq(model1.fit_predict([data])[0], model2.fit_predict([data])[0])


def test_kcenters_6():
    # test with a custom metric when the input data isn't a list of numpy arrays

    x = md.Trajectory(xyz=np.random.randn(100,1,3), topology=None)
    # just get the sqeuclidean for the first atom along the first coordinate
    metric = lambda target, ref, i: (target.xyz[:, 0, 0] - ref.xyz[i, 0, 0])**2
    model1 = KCenters(n_clusters=10, metric=metric, random_state=0)
    model1.fit([x])

    model2 = KCenters(n_clusters=10, metric='sqeuclidean', random_state=0)
    model2.fit([x.xyz[:, :, 0]])
    eq(reduce(operator.add, model1.cluster_centers_).xyz[:, 0, 0],
       model2.cluster_centers_[:, 0])


def test_kcenters_7():
    # are fit_predict and fit().predict() consistent?
    for opt in [True, False]:
        trj = np.random.RandomState(0).randn(30, 2)
        k = KCenters(n_clusters=10, random_state=0, opt=opt).fit([trj])
        l1 = KCenters(n_clusters=10, random_state=0, opt=opt).fit([trj]).predict([trj])[0]
        l2 = KCenters(n_clusters=10, random_state=0, opt=opt).fit_predict([trj])[0]

        eq(l1, l2)


def test_kcenters_8():
    X = np.random.RandomState(1).randn(100, 2)
    for dtype in [np.float64, np.float32]:
        X = X.astype(dtype)
        m1 = KCenters(n_clusters=10, random_state=0, opt=True).fit([X])
        m2 = KCenters(n_clusters=10, random_state=0, opt=False).fit([X])

        eq(m1.cluster_centers_, m2.cluster_centers_)
        eq(m1.distances_[0], m2.distances_[0])
        eq(m1.labels_[0], m2.labels_[0])
        assert np.all(np.logical_not(np.isnan(m1.distances_[0])))
        eq(m1.predict([X])[0], m2.predict([X])[0])
        eq(m1.predict([X])[0], m1.labels_[0])
