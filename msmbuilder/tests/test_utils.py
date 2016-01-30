from __future__ import division

import numpy as np
import sklearn.pipeline
from mdtraj.testing import eq
from sklearn.externals.joblib import dump as jl_dump

from msmbuilder.decomposition import tICA
from msmbuilder.utils import Subsampler, dump, load
from .test_commands import tempdir

random = np.random.RandomState(2)


def test_subsampler_lag1():
    n_traj, n_samples, n_features = 3, 100, 7
    lag_time = 1
    X_all_0 = [random.normal(size=(n_samples, n_features))
               for i in range(n_traj)]
    q_0 = np.concatenate(X_all_0)

    subsampler = Subsampler(lag_time=lag_time)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)

    eq(q_0.shape, q_1.shape)
    eq(q_0.mean(0), q_1.mean(0))
    eq(q_0.std(0), q_1.std(0))

    subsampler = Subsampler(lag_time=lag_time, sliding_window=False)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)

    eq(q_0.shape, q_1.shape)
    eq(q_0.mean(0), q_1.mean(0))
    eq(q_0.std(0), q_1.std(0))


def test_subsampler_lag2():
    n_traj, n_samples, n_features = 3, 100, 7
    lag_time = 2
    X_all_0 = [random.normal(size=(n_samples, n_features))
               for i in range(n_traj)]
    q_0 = np.concatenate(X_all_0)

    subsampler = Subsampler(lag_time=lag_time)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)

    eq(((n_samples - lag_time + 2) * n_traj, n_features), q_1.shape)

    subsampler = Subsampler(lag_time=lag_time, sliding_window=False)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)

    eq(((n_samples / lag_time) * n_traj, n_features), q_1.shape)


def test_subsampler_tica():
    n_traj, n_samples, n_features = 1, 500, 4
    lag_time = 2
    X_all_0 = [random.normal(size=(n_samples, n_features))
               for i in range(n_traj)]
    tica_0 = tICA(lag_time=lag_time)
    tica_0.fit(X_all_0)

    subsampler = Subsampler(lag_time=lag_time)
    tica_1 = tICA()
    pipeline = sklearn.pipeline.Pipeline([
        ("subsampler", subsampler),
        ('tica', tica_1)
    ])
    pipeline.fit(X_all_0)

    eq(tica_0.n_features, tica_1.n_features)  # Obviously true
    eq(tica_0.n_observations_, tica_1.n_observations_)
    # The eigenvalues should be the same.  NOT the timescales,
    # as tica_1 has timescales calculated in a different time unit
    eq(tica_0.eigenvalues_, tica_1.eigenvalues_)


def test_dump_load():
    data = dict(name="Fancy_name", arr=np.random.rand(10, 5))
    with tempdir():
        dump(data, 'filename')
        data2 = load('filename')
    eq(data, data2)


def test_load_legacy():
    # Used to save joblib files
    data = dict(name="Fancy_name", arr=np.random.rand(10, 5))
    with tempdir():
        jl_dump(data, 'filename', compress=1)
        data2 = load('filename')
    eq(data, data2)
