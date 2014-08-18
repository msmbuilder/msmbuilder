from __future__ import print_function, division
import os
import numpy as np
from numpy.testing import assert_approx_equal
from mdtraj.testing import eq
import scipy.sparse
from sklearn.externals.joblib import load, dump
import sklearn.pipeline
from mixtape import cluster
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.utils import map_drawn_samples
import mdtraj as md
import pandas as pd
from six import PY3


def test_1():
    # test counts matrix without trimming
    model = MarkovStateModel(reversible_type=None, ergodic_cutoff=0)

    model.fit([[1,1,1,1,1,1,1,1,1]])
    eq(model.countsmat_, np.array([[8.0]]))
    eq(model.mapping_, {1: 0})


def test_2():
    # test counts matrix with trimming
    model = MarkovStateModel(reversible_type=None, ergodic_cutoff=1)

    model.fit([[1,1,1,1,1,1,1,1,1, 2]])
    eq(model.mapping_, {1: 0})
    eq(model.countsmat_, np.array([[8]]))

def test_3():
    model = MarkovStateModel(reversible_type='mle')
    model.fit([[0,0,0,0,1,1,1,1,0,0,0,0,2,2,2,2,0,0,0]])

    counts = np.array([[8, 1, 1], [1, 3, 0], [1, 0, 3]])
    eq(model.countsmat_, counts)
    assert np.sum(model.populations_) == 1.0
    model.timescales_

    # test pickleable
    try:
        dump(model, 'test-msm-temp.npy', compress=1)
        model2 = load('test-msm-temp.npy')
        eq(model2.timescales_, model.timescales_)
    finally:
        os.unlink('test-msm-temp.npy')


def test_4():
    data = [np.random.randn(10, 1), np.random.randn(100, 1)]
    print(cluster.KMeans(n_clusters=3).fit_predict(data))
    print(cluster.MiniBatchKMeans(n_clusters=3).fit_predict(data))
    print(cluster.AffinityPropagation().fit_predict(data))
    print(cluster.MeanShift().fit_predict(data))
    print(cluster.SpectralClustering(n_clusters=2).fit_predict(data))
    print(cluster.Ward(n_clusters=2).fit_predict(data))

def test_5():
    # test score_ll
    model = MarkovStateModel(reversible_type='mle')
    sequence = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
    model.fit([sequence])
    assert model.mapping_ == {'a': 0, 'b': 1}

    score_aa = model.score_ll([['a', 'a']])
    assert score_aa == np.log(model.transmat_[0,0])
    score_bb = model.score_ll([['b', 'b']])
    assert score_bb == np.log(model.transmat_[1,1])
    score_ab = model.score_ll([['a', 'b']])
    assert score_ab == np.log(model.transmat_[0,1])
    score_abb =  model.score_ll([['a', 'b', 'b']])
    assert score_abb == np.log(model.transmat_[0,1]) + np.log(model.transmat_[1,1])

    assert model.state_labels_ == ['a', 'b']
    assert np.sum(model.populations_) == 1.0

def test_51():
    # test score_ll
    model = MarkovStateModel(reversible_type='mle')
    sequence = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'a', 'a']
    model.fit([sequence])
    assert model.mapping_ == {'a': 0, 'b': 1, 'c': 2}

    score_ac = model.score_ll([['a', 'c']])
    assert score_ac == np.log(model.transmat_[0,2])

def test_6():
    # test score_ll with novel entries
    model = MarkovStateModel(reversible_type='mle')
    sequence = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
    model.fit([sequence])

    assert not np.isfinite(model.score_ll([['c']]))
    assert not np.isfinite(model.score_ll([['c', 'c']]))
    assert not np.isfinite(model.score_ll([['a', 'c']]))

def test_7():
    # test timescales
    model = MarkovStateModel()
    model.fit([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]])
    assert np.all(np.isfinite(model.timescales_))
    assert len(model.timescales_) == 1

    model.fit([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]])
    assert np.all(np.isfinite(model.timescales_))
    assert len(model.timescales_) == 2
    assert model.n_states_ == 3

    model = MarkovStateModel(n_timescales=1)
    model.fit([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]])
    assert len(model.timescales_) == 1

    model = MarkovStateModel(n_timescales=100)
    model.fit([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]])
    assert len(model.timescales_) == 2
    assert np.sum(model.populations_) == 1.0

def test_8():
    # test transform
    model = MarkovStateModel()
    model.fit([['a', 'a', 'b', 'b', 'c', 'c', 'a', 'a']])
    assert model.mapping_ == {'a': 0, 'b': 1, 'c': 2}

    v = model.transform([['a', 'b', 'c']])
    assert isinstance(v, list)
    assert len(v) == 1
    assert v[0].dtype == np.int
    np.testing.assert_array_equal(v[0], [0, 1, 2])

    v = model.transform([['a', 'b', 'c', 'd']], 'clip')
    assert isinstance(v, list)
    assert len(v) == 1
    assert v[0].dtype == np.int
    np.testing.assert_array_equal(v[0], [0, 1, 2])

    v = model.transform([['a', 'b', 'c', 'd']], 'clip')
    assert isinstance(v, list)
    assert len(v) == 1
    assert v[0].dtype == np.int
    np.testing.assert_array_equal(v[0], [0, 1, 2])

    v = model.transform([['a', 'b', 'c', 'd']], 'fill')
    assert isinstance(v, list)
    assert len(v) == 1
    assert v[0].dtype == np.float
    np.testing.assert_array_equal(v[0], [0, 1, 2, np.nan])

    v = model.transform([['a', 'a', 'SPLIT', 'b', 'b', 'b']], 'clip')
    assert isinstance(v, list)
    assert len(v) == 2
    assert v[0].dtype == np.int
    assert v[1].dtype == np.int
    np.testing.assert_array_equal(v[0], [0, 0])
    np.testing.assert_array_equal(v[1], [1, 1, 1])


def test_9():
    # what if the input data contains NaN? They should be ignored
    model = MarkovStateModel(ergodic_cutoff=0)

    seq = [0, 1, 0, 1, np.nan]
    model.fit(seq)
    assert model.n_states_ == 2
    assert model.mapping_ == {0:0, 1:1}

    if not PY3:
        model = MarkovStateModel()
        seq = [0, 1, 0, None, 0, 1]
        model.fit(seq)
        assert model.n_states_ == 2
        assert model.mapping_ == {0:0, 1:1}

def test_10():
    # test inverse transform
    model = MarkovStateModel(reversible_type=None, ergodic_cutoff=0)
    model.fit([['a', 'b', 'c', 'a', 'a', 'b']])
    v = model.inverse_transform([[0, 1, 2]])
    assert len(v) == 1
    np.testing.assert_array_equal(v[0], ['a', 'b', 'c'])

def test_11():
    # test sample
    model = MarkovStateModel()
    model.fit([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]])
    sample = model.sample(n_steps=1000, random_state=0)
    assert isinstance(sample, np.ndarray)
    assert len(sample) == 1000

    bc = np.bincount(sample)
    diff = model.populations_ - (bc / np.sum(bc))

    assert np.sum(np.abs(diff)) < 0.1

def test_12():
    # test eigtransform
    model = MarkovStateModel(n_timescales=1)
    model.fit([[4, 3, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]])
    assert model.mapping_ == {0:0, 1:1, 2:2}
    assert len(model.eigenvalues_) == 2
    t = model.eigtransform([[0, 1]], right=True)
    assert t[0][0] == model.right_eigenvectors_[0, 1]
    assert t[0][1] == model.right_eigenvectors_[1, 1]

    s = model.eigtransform([[0, 1]], right=False)
    assert s[0][0] == model.left_eigenvectors_[0, 1]
    assert s[0][1] == model.left_eigenvectors_[1, 1]


def test_eigtransform_2():
    model = MarkovStateModel(n_timescales=2)
    traj = [4, 3, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0]
    model.fit([traj])

    transformed_0 = model.eigtransform([traj], mode='clip')
    # clip off the first two states (not ergodic)
    assert transformed_0[0].shape == (len(traj)-2, model.n_timescales)

    transformed_1 = model.eigtransform([traj], mode='fill')
    assert transformed_1[0].shape == (len(traj), model.n_timescales)
    assert np.all(np.isnan(transformed_1[0][:2, :]))
    assert not np.any(np.isnan(transformed_1[0][2:]))

def test_13():
    model = MarkovStateModel(n_timescales=2)
    model.fit([[0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1, 2, 2, 0, 0]])
    left_right = np.dot(model.left_eigenvectors_.T, model.right_eigenvectors_)

    # check biorthonormal
    np.testing.assert_array_almost_equal(
        left_right,
        np.eye(3))

    # check that the stationary left eigenvector is normalized to be 1
    np.testing.assert_almost_equal(model.left_eigenvectors_[:, 0].sum(), 1)

    # the left eigenvectors satisfy <\phi_i, \phi_i>_{\mu^{-1}} = 1
    for i in range(3):
        np.testing.assert_almost_equal(
            np.dot(model.left_eigenvectors_[:, i], model.left_eigenvectors_[:, i] /
            model.populations_), 1)

    # and that the right eigenvectors satisfy  <\psi_i, \psi_i>_{\mu} = 1
    for i in range(3):
        np.testing.assert_almost_equal(
            np.dot(model.right_eigenvectors_[:, i], model.right_eigenvectors_[:, i] *
            model.populations_), 1)


def test_14():
    from mixtape.datasets import load_doublewell
    from mixtape.cluster import NDGrid
    from sklearn.pipeline import Pipeline

    ds = load_doublewell(random_state=0)

    p = Pipeline([
        ('ndgrid', NDGrid(n_bins_per_feature=100)),
        ('msm', MarkovStateModel(lag_time=100))
    ])

    p.fit(ds.trajectories)
    p.named_steps['msm'].summary()


def test_sample_1():
    # Test that the code actually runs and gives something non-crazy
    # Make an ergodic dataset with two gaussian centers offset by 25 units.
    chunk = np.random.normal(size=(20000, 3))
    data = [np.vstack((chunk, chunk + 25)), np.vstack((chunk + 25, chunk))]

    clusterer = cluster.KMeans(n_clusters=2)
    msm = MarkovStateModel()
    pipeline = sklearn.pipeline.Pipeline([("clusterer", clusterer), ("msm", msm)])
    pipeline.fit(data)
    trimmed_assignments = pipeline.transform(data)
    
    # Now let's make make the output assignments start with zero at the first position.
    i0 = trimmed_assignments[0][0]
    if i0 == 1:
        for m in trimmed_assignments:
            m *= -1
            m += 1
    
    pairs = msm.draw_samples(trimmed_assignments, 2000)

    samples = map_drawn_samples(pairs, data)
    mu = np.mean(samples, axis=1)
    eq(mu, np.array([[0., 0., 0.0], [25., 25., 25.]]), decimal=1)

    # We should make sure we can sample from Trajectory objects too...
    # Create a fake topology with 1 atom to match our input dataset
    top = md.Topology.from_dataframe(pd.DataFrame({"serial":[0], "name":["HN"], "element":["H"], "resSeq":[1], "resName":"RES", "chainID":[0]}), bonds=np.zeros(shape=(0, 2), dtype='int'))
    trajectories = [md.Trajectory(x[:, np.newaxis], top) for x in data]  # np.newaxis reshapes the data to have a 40000 frames, 1 atom, 3 xyz

    trj_samples = map_drawn_samples(pairs, trajectories)
    mu = np.array([t.xyz.mean(0)[0] for t in trj_samples])
    eq(mu, np.array([[0., 0., 0.0], [25., 25., 25.]]), decimal=1)


def test_score_1():
    # test that GMRQ is equal to the sum of the first n eigenvalues,
    # when testing and training on the same dataset.
    sequence = [0,0,0,1,1,1,2,2,2,1,1,1,0,0,0,1,2,2,2,1,1,1,0,0]
    for n in [0, 1, 2]:
        model = MarkovStateModel(verbose=False, n_timescales=n)
        model.fit([sequence])

        assert_approx_equal(
            model.score([sequence]),
            model.eigenvalues_.sum())
