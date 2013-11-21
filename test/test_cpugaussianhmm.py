import numpy as np
from scipy.misc import logsumexp
from sklearn.hmm import GaussianHMM
from mixtape._cpuhmm import CPUGaussianHMM

def test_1():
    "Test the getters and setters"
    t1 = np.random.randn(10, 2)
    n_features = 2
    for n_states in [3, 4]:
        hmm = CPUGaussianHMM(n_states, n_features)
        hmm._sequences = [t1]
        
        means = np.random.randn(n_states, n_features)
        hmm.means_ = means
        yield lambda: np.testing.assert_array_almost_equal(hmm.means_, means)

        variances = np.random.rand(n_states, n_features)
        hmm.variances_ = variances
        yield lambda: np.testing.assert_array_almost_equal(hmm.variances_, variances)

        transmat = np.random.rand(n_states, n_states)
        hmm.transmat_ = transmat
        yield lambda: np.testing.assert_array_almost_equal(hmm.transmat_, transmat)

        startprob = np.random.rand(n_states)
        hmm.startprob_ = startprob
        yield lambda: np.testing.assert_array_almost_equal(hmm.startprob_, startprob)

def test_2():
    n_features = 3
    length = 1000
    
    for n_states in [4]:
        t1 = np.random.randn(length, n_features)
        means = np.random.randn(n_states, n_features)
        variances = np.random.rand(n_states, n_features)
        transmat = np.random.rand(n_states, n_states)
        transmat = transmat / np.sum(transmat, axis=1)[:, None]
        startprob = np.random.rand(n_states)
        startprob = startprob / np.sum(startprob)
        
        chmm = CPUGaussianHMM(n_states, n_features)
        chmm._sequences = [t1]

        pyhmm = GaussianHMM(n_components=n_states, init_params='', params='', covariance_type='diag')
        chmm.means_ = means
        chmm.variances_ = variances
        chmm.transmat_ = transmat
        chmm.startprob_ = startprob
        cstats = chmm._do_estep()

        pyhmm.means_ = means
        pyhmm.covars_ = variances
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

        print '\nrefence'
        print stats['trans']
        print 'kernel'
        print cstats['trans']

        yield lambda: np.testing.assert_array_almost_equal(stats['trans'], cstats['trans'], decimal=4)
        yield lambda: np.testing.assert_array_almost_equal(stats['post'], cstats['post'], decimal=4)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs'], cstats['obs'], decimal=4)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs**2'], cstats['obs**2'], decimal=4)
