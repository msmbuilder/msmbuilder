from __future__ import print_function, division, absolute_import

import numpy  as np
from scipy.misc import logsumexp
from sklearn.hmm import GaussianHMM
from nose.plugins.attrib import attr


@attr('cuda')
def test_1():
    "Test the getters and setters, which transfer data to/from the GPU"
    # from mixtape._cuda_ghmm_mixed import GaussianHMMCUDAImpl
    
    t1 = np.random.randn(10, 2).astype(np.float32)
    n_features = 2
    for n_states in [3, 4]:
        hmm = GaussianHMMCUDAImpl(n_states, n_features)
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

@attr('cuda')
def test_2():
    # from mixtape._cuda_ghmm_mixed import GaussianHMMCUDAImpl
    
    np.random.seed(42)
    n_features = 32
    length = 20

    #for n_states in [3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32]:
    for n_states in [8]:
        t1 = np.random.randn(length, n_features)
        means = np.random.randn(n_states, n_features)
        vars = np.random.rand(n_states, n_features)
        transmat = np.random.rand(n_states, n_states)
        transmat = transmat / np.sum(transmat, axis=1)[:, None]
        startprob = np.random.rand(n_states)
        startprob = startprob / np.sum(startprob)

        cuhmm = GaussianHMMCUDAImpl(n_states, n_features)
        cuhmm._sequences = [t1]

        pyhmm = GaussianHMM(n_components=n_states, init_params='', params='', covariance_type='diag')
        cuhmm.means_ = means
        cuhmm.vars_ = vars
        cuhmm.transmat_ = transmat
        cuhmm.startprob_ = startprob
        logprob, custats = cuhmm.do_estep()

        pyhmm.means_ = means
        pyhmm.covars_ = vars
        pyhmm.transmat_ = transmat
        pyhmm.startprob_ = startprob
        pyhmm._initialize_sufficient_statistics()

        framelogprob = pyhmm._compute_log_likelihood(t1)
        cuframelogprob = cuhmm._get_framelogprob()
        yield lambda: np.testing.assert_array_almost_equal(framelogprob, cuframelogprob, decimal=3)

        fwdlattice = pyhmm._do_forward_pass(framelogprob)[1]
        cufwdlattice = cuhmm._get_fwdlattice()
        yield lambda: np.testing.assert_array_almost_equal(fwdlattice, cufwdlattice, decimal=3)

        bwdlattice = pyhmm._do_backward_pass(framelogprob)
        cubwdlattice = cuhmm._get_bwdlattice()
        yield lambda: np.testing.assert_array_almost_equal(bwdlattice, cubwdlattice, decimal=3)

 
        gamma = fwdlattice + bwdlattice
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        cuposteriors = cuhmm._get_posteriors()
        yield lambda: np.testing.assert_array_almost_equal(posteriors, cuposteriors, decimal=3)

        stats = pyhmm._initialize_sufficient_statistics()
        pyhmm._accumulate_sufficient_statistics(
            stats, t1, framelogprob, posteriors, fwdlattice,
            bwdlattice, 'stmc')

        print('ref transcounts')
        print(transitioncounts(cufwdlattice, cubwdlattice, cuframelogprob, np.log(transmat)))
        print('cutranscounts')
        print(ustats['trans'])

        yield lambda: np.testing.assert_array_almost_equal(stats['trans'], custats['trans'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['post'], custats['post'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs'], custats['obs'], decimal=3)
        yield lambda: np.testing.assert_array_almost_equal(stats['obs**2'], custats['obs**2'], decimal=3)



def reference_forward(framelogprob, startprob, transmat):
    log_transmat = np.log(transmat)
    log_startprob = np.log(startprob)
    fwdlattice = np.zeros_like(framelogprob)
    n_observations, n_components = framelogprob.shape
    work_buffer = np.zeros(n_components)

    for i in range(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

    for t in range(1, n_observations):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]
            print("reference: t=%d, j=%d, logsumexp=%f, flp=%f, sum=%f    workbuffer=%s" % (t, j, logsumexp(work_buffer), framelogprob[t, j], logsumexp(work_buffer) + framelogprob[t, j], str(work_buffer)))
            fwdlattice[t, j] = logsumexp(work_buffer) + framelogprob[t, j]
    return fwdlattice



def reference_backward(framelogprob, startprob, transmat):
    log_transmat = np.log(transmat)
    log_startprob = np.log(startprob)
    bwdlattice = np.zeros_like(framelogprob)
    n_observations, n_components = framelogprob.shape
    work_buffer = np.zeros(n_components)

    for i in range(n_components):
        bwdlattice[n_observations - 1, i] = 0.0

    for t in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = log_transmat[i, j] + framelogprob[t + 1, j] + bwdlattice[t + 1, j]

                if (i == 0):
                    print('log_transmat[i, %d]    '%j, log_transmat[i, j])
                    print('framelogprob[t + 1, %d]'%j,  framelogprob[t + 1, j])
                    print('bwdlattice[t + 1, %d]  '%j, bwdlattice[t + 1, j])
            if i == 0:
                print('bwd[%d, %d] = logsumexp(%s) = %f' % (t, i, str(work_buffer), logsumexp(work_buffer)))
            bwdlattice[t, i] = logsumexp(work_buffer)

    return bwdlattice


def transitioncounts(fwdlattice, bwdlattice, framelogprob, log_transmat):
    n_observations, n_components = fwdlattice.shape
    lneta = np.zeros((n_observations-1, n_components, n_components))
    from scipy.misc import logsumexp
    logprob = logsumexp(fwdlattice[n_observations-1, :])
    print('logprob', logprob)

    for t in range(n_observations - 1):
        for i in range(n_components):
            for j in range(n_components):
                lneta[t, i, j] = fwdlattice[t, i] + log_transmat[i, j] \
                    + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - logprob


    print(framelogprob)

    print('fwdlattice[:, 0]')
    print(fwdlattice[:, 0])
    print('logtransmat[0,0]')
    print(log_transmat[0,0])
    print('framelogprob')
    print(framelogprob[:, 0])
    print('bwdlattice[:, 0]')
    print(bwdlattice[:, 0])
    print('lneta{0,0}')
    print(lneta[:, 0, 0])

    return np.exp(logsumexp(lneta, 0))
