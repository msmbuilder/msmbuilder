import numpy as np
from mdtraj.testing import eq
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score

from msmbuilder.cluster import LandmarkAgglomerative
from msmbuilder.example_datasets import AlanineDipeptide

random = np.random.RandomState(2)


def test_1():
    x = [random.randn(10, 2), random.randn(10, 2)]

    n_clusters = 2
    model1 = LandmarkAgglomerative(n_clusters=n_clusters)
    model2 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   n_landmarks=sum(len(s) for s in x))

    labels0 = clone(model1).fit(x).predict(x)
    labels1 = model1.fit_predict(x)
    labels2 = model2.fit_predict(x)

    assert len(labels0) == 2
    assert len(labels1) == 2
    assert len(labels2) == 2
    eq(labels0[0], labels1[0])
    eq(labels0[1], labels1[1])
    eq(labels0[0], labels2[0])
    eq(labels0[1], labels2[1])

    assert len(np.unique(np.concatenate(labels0))) == n_clusters


def test_2():
    # this should be a really easy clustering problem
    x = [random.randn(20, 2) + 10, random.randn(20, 2)]

    n_clusters = 2
    model1 = LandmarkAgglomerative(n_clusters=n_clusters)
    model2 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   landmark_strategy='random',
                                   random_state=random, n_landmarks=20)

    labels1 = model1.fit_predict(x)
    labels2 = model2.fit_predict(x)
    assert adjusted_rand_score(np.concatenate(labels1),
                               np.concatenate(labels2)) == 1.0


def test_callable_metric():
    def my_euc(target, ref, i):
        return np.sqrt(np.sum((target - ref[i]) ** 2, axis=1))

    model1 = LandmarkAgglomerative(n_clusters=10, n_landmarks=20,
                                   metric='euclidean')
    model2 = LandmarkAgglomerative(n_clusters=10, n_landmarks=20, metric=my_euc)

    data = np.random.RandomState(0).randn(100, 2)
    eq(model1.fit_predict([data])[0], model2.fit_predict([data])[0])


def test_1_ward():
    x = [random.randn(10, 2), random.randn(10, 2)]

    n_clusters = 2
    model1 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   linkage='ward')
    model2 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   linkage='ward',
                                   n_landmarks=sum(len(s) for s in x))

    labels0 = clone(model1).fit(x).predict(x)
    labels1 = model1.fit_predict(x)
    labels2 = model2.fit_predict(x)

    assert len(labels0) == 2
    assert len(labels1) == 2
    assert len(labels2) == 2
    eq(labels0[0], labels1[0])
    eq(labels0[1], labels1[1])
    eq(labels0[0], labels2[0])
    eq(labels0[1], labels2[1])

    assert len(np.unique(np.concatenate(labels0))) == n_clusters


def test_2_ward():
    # this should be a really easy clustering problem
    x = [random.randn(20, 2) + 10, random.randn(20, 2)]

    n_clusters = 2
    model1 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   linkage='ward')
    model2 = LandmarkAgglomerative(n_clusters=n_clusters,
                                   linkage='ward',
                                   landmark_strategy='random',
                                   random_state=random, n_landmarks=20)

    labels1 = model1.fit_predict(x)
    labels2 = model2.fit_predict(x)
    assert adjusted_rand_score(np.concatenate(labels1),
                               np.concatenate(labels2)) == 1.0


def test_alanine_dipeptide():
    # test for rmsd metric compatibility with ward clustering
    # keep n_landmarks small or this will get really slow
    trajectories = AlanineDipeptide().get_cached().trajectories
    n_clusters = 4
    model = LandmarkAgglomerative(n_clusters=n_clusters, n_landmarks=20,
                                  linkage='ward', metric='rmsd')
    labels = model.fit_predict(trajectories[0][0:100])

    assert len(np.unique(np.concatenate(labels))) <= n_clusters
