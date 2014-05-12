from __future__ import print_function
import os
import numpy as np
from mdtraj.testing import eq
import scipy.sparse
from sklearn.externals.joblib import load, dump
from mixtape import cluster
from mixtape.markovstatemodel import MarkovStateModel

def todense(mat):
    if scipy.sparse.issparse(mat):
        return np.asarray(mat.todense())
    return mat

def test_1():
    # test counts matrix without trimming
    model = MarkovStateModel(n_states=2, reversible_type=None, ergodic_trim=False)

    model.fit([[1,1,1,1,1,1,1,1,1]])
    eq(todense(model.countsmat_), np.array([[0, 0], [0, 8]]))

def test_2():
    # test counts matrix with trimming
    model = MarkovStateModel(n_states=2, reversible_type=None, ergodic_trim=True)

    model.fit([[1,1,1,1,1,1,1,1,1]])
    eq(model.mapping_, {1: 0})
    eq(todense(model.countsmat_), np.array([[8]]))

def test_3():
    model = MarkovStateModel(n_states=3, reversible_type='mle', ergodic_trim=True)
    model.fit([[0,0,0,0,1,1,1,1,0,0,0,0,2,2,2,2,0,0,0]])

    counts = np.array([[8, 1, 1], [1, 3, 0], [1, 0, 3]])
    eq(todense(model.rawcounts_), counts)
    eq(todense(model.countsmat_), counts)
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
