from __future__ import print_function
import os
import numpy as np
from mdtraj.testing import eq
import scipy.sparse
from sklearn.externals.joblib import load, dump
import sklearn.pipeline
from mixtape import cluster
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.utils import map_drawn_samples
import mdtraj as md
import pandas as pd

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


def test_sample_1():  # Test that the code actually runs and gives something non-crazy
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
