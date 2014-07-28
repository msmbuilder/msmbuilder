from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from sklearn.hmm import GaussianHMM
from mixtape._ghmm import GaussianHMMCPUImpl

def test_1():
    "Test the getters and setters"
    t1 = np.random.randn(10, 2)
    n_features = 2
    for n_states in [3, 4]:
        hmm = GaussianHMMCPUImpl(n_states, n_features)
        hmm._sequences = [t1]
        
        means = np.random.randn(n_states, n_features).astype(np.float32)
        hmm.means_ = means
        yield lambda: np.testing.assert_array_almost_equal(hmm.means_, means)

        vars = np.random.rand(n_states, n_features).astype(np.float32)
        hmm.vars_ = vars
        yield lambda: np.testing.assert_array_almost_equal(hmm.vars_, vars)

        transmat = np.random.rand(n_states, n_states).astype(np.float32)
        hmm.transmat_ = transmat
        yield lambda: np.testing.assert_array_almost_equal(hmm.transmat_, transmat)

        startprob = np.random.rand(n_states).astype(np.float32)
        hmm.startprob_ = startprob
        yield lambda: np.testing.assert_array_almost_equal(hmm.startprob_, startprob)

def test_2():
    n_features = 3
    length = 32
    
    for n_states in [4]:
        t1 = np.random.randn(length, n_features)
        means = np.random.randn(n_states, n_features)
        vars = np.random.rand(n_states, n_features)
        transmat = np.random.rand(n_states, n_states)
        transmat = transmat / np.sum(transmat, axis=1)[:, None]
        startprob = np.random.rand(n_states)
        startprob = startprob / np.sum(startprob)
        
        chmm = GaussianHMMCPUImpl(n_states, n_features)
        chmm._sequences = [t1]

        pyhmm = GaussianHMM(n_components=n_states, init_params='', params='', covariance_type='diag')
        chmm.means_ = means.astype(np.float32)
        chmm.vars_ = vars.astype(np.float32)
        chmm.transmat_ = transmat.astype(np.float32)
        chmm.startprob_ = startprob.astype(np.float32)
        clogprob, cstats = chmm.do_estep()

        pyhmm.means_ = means
        pyhmm.covars_ = vars
        pyhmm.transmat_ = transmat
        pyhmm.startprob_ = startprob

        framelogprob = pyhmm._compute_log_likelihood(t1)
        fwdlattice = pyhmm._do_forward_pass(framelogprob)[1]
        bwdlattice = pyhmm._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        stats = pyhmm._initialize_sufficient_statistics()
        pyhmm._accumulate_sufficient_statistics(
            stats, t1, framelogprob, posteriors, fwdlattice,
            bwdlattice, 'stmc')

        yield lambda: np.testing.assert_array_almost_equal(stats['trans'], cstats['trans'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['post'], cstats['post'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs'], cstats['obs'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs**2'], cstats['obs**2'], decimal=3)
        
