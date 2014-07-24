from __future__ import print_function
import os
import numpy as np
from mdtraj.testing import eq
import scipy.sparse
from sklearn.externals.joblib import load, dump
from mixtape import cluster
from mixtape.markovstatemodel import MarkovStateModel


def test_1():
    # test counts matrix without trimming
    model = MarkovStateModel(reversible_type=None, ergodic_trim=False)

    model.fit([[1,1,1,1,1,1,1,1,1]])
    eq(model.countsmat_, np.array([[8.0]]))
    eq(model.mapping_, {1: 0})

def test_2():
    # test counts matrix with trimming
    model = MarkovStateModel(reversible_type=None, ergodic_trim=True)

    model.fit([[1,1,1,1,1,1,1,1,1, 2]])
    eq(model.mapping_, {1: 0})
    eq(model.countsmat_, np.array([[8]]))

def test_3():
    model = MarkovStateModel(reversible_type='mle', ergodic_trim=True)
    model.fit([[0,0,0,0,1,1,1,1,0,0,0,0,2,2,2,2,0,0,0]])

    counts = np.array([[8, 1, 1], [1, 3, 0], [1, 0, 3]])
    eq(model.countsmat_, counts)
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
    model = MarkovStateModel(reversible_type='mle', ergodic_trim=True)
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

def test_51():
    # test score_ll
    model = MarkovStateModel(reversible_type='mle', ergodic_trim=True)
    sequence = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'a', 'a']
    model.fit([sequence])
    assert model.mapping_ == {'a': 0, 'b': 1, 'c': 2}

    score_ac = model.score_ll([['a', 'c']])
    assert score_ac == np.log(model.transmat_[0,2])

def test_6():
    # test score_ll with novel entries
    model = MarkovStateModel(reversible_type='mle', ergodic_trim=True)
    sequence = ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']
    model.fit([sequence])

    assert not np.isfinite(model.score_ll([['c']]))
    assert not np.isfinite(model.score_ll([['c', 'c']]))
    assert not np.isfinite(model.score_ll([['a', 'c']]))
    