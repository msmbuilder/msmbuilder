import numpy as np
from mixtape._cpuhmm import CPUGaussianHMM

def test_1():
    n_states = 4
    n_features = 3
    t = np.random.randn(100, n_features)
    chmm = CPUGaussianHMM(n_states, n_features)
    means = np.random.randn(n_states, n_features)
    variances = np.random.rand(n_states, n_features)
    transmat = np.random.rand(n_states, n_states)
    transmat = transmat / np.sum(transmat, axis=1)[:, None]
    startprob = np.random.rand(n_states)
    startprob = startprob / np.sum(startprob)
    chmm.means_ = means
    chmm.variances_ = variances
    chmm.transmat_ = transmat
    chmm.startprob_ = startprob
    chmm.fit([t])
