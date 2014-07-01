import os
import numpy as np
import mdtraj as md
from mdtraj.testing import eq
import mixtape

random = np.random.RandomState(2)

def test_subsampler_lag1():
    n_traj, n_samples, n_features = 3, 100, 7
    lag_time = 1
    X_all_0 = [random.normal(size=(n_samples, n_features)) for i in range(n_traj)]
    q_0 = np.concatenate(X_all_0)

    subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)
    
    eq(q_0.shape, q_1.shape)
    eq(q_0.mean(0), q_1.mean(0))
    eq(q_0.std(0), q_1.std(0))

    subsampler = mixtape.utils.Subsampler(lag_time=lag_time, sliding_window=False)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)
    
    eq(q_0.shape, q_1.shape)
    eq(q_0.mean(0), q_1.mean(0))
    eq(q_0.std(0), q_1.std(0))


def test_subsampler_lag2():
    n_traj, n_samples, n_features = 3, 100, 77
    lag_time = 2
    X_all_0 = [random.normal(size=(n_samples, n_features)) for i in range(n_traj)]
    q_0 = np.concatenate(X_all_0)

    subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)
    
    eq(((n_samples - lag_time + 2) * n_traj, n_features), q_1.shape)

    subsampler = mixtape.utils.Subsampler(lag_time=lag_time, sliding_window=False)
    X_all_1 = subsampler.transform(X_all_0)
    q_1 = np.concatenate(X_all_1)
    
    eq(((n_samples / lag_time) * n_traj, n_features), q_1.shape)
