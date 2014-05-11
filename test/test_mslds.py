'''Currently the mslds code is basically a Gaussian HMM with a full-rank covariance
matrix with some extra bells and whisles. But, because of the way the E-step works
currently, the means and covariances are estimated exactly as with a Gaussian HMM.

Then, afterwards, the A, b and Q are estimated. So, we can do a lot of testing
by comparing to a reference gaussian HMM implementation
'''
import string
import numpy as np
from sklearn.hmm import GaussianHMM
from sklearn.utils.extmath import logsumexp
from mixtape.mslds import MetastableSwitchingLDS
from mixtape import _mslds

N_STATES = 2
data = [np.random.randn(100, 3), np.random.randn(100, 3)]
refmodel = GaussianHMM(n_components=N_STATES, covariance_type='full').fit(data)

def _sklearn_estep():
    # copied from sklearn/hmm.py#L440
    curr_logprob = 0
    stats = refmodel._initialize_sufficient_statistics()
    stats['post[1:]'] = np.zeros(refmodel.n_components)
    stats['post[:-1]'] = np.zeros(refmodel.n_components)
    stats['obs[1:]'] = np.zeros((refmodel.n_components,
                                    refmodel.n_features))
    stats['obs[:-1]'] = np.zeros((refmodel.n_components,
                                    refmodel.n_features))
    stats['obs*obs[t-1].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))
    stats['obs[1:]*obs[1:].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))
    stats['obs[:-1]*obs[:-1].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))

    for seq in data:
        framelogprob = refmodel._compute_log_likelihood(seq)
        lpr, fwdlattice = refmodel._do_forward_pass(framelogprob)
        bwdlattice = refmodel._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        curr_logprob += lpr
        refmodel._accumulate_sufficient_statistics(stats, seq,
                            framelogprob, posteriors, fwdlattice,
                            bwdlattice, string.ascii_letters)

        # accumulate the extra stats that our model has but the
        # sklearn model doesn't
        stats['post[1:]'] += posteriors[1:].sum(axis=0)
        stats['post[:-1]'] += posteriors[:-1].sum(axis=0)
        stats['obs[1:]'] += np.dot(posteriors[1:].T, seq[1:])
        stats['obs[:-1]'] += np.dot(posteriors[:-1].T, seq[:-1])

        for t in range(1, len(seq)):
            obsobsTminus1 = np.outer(seq[t], seq[t-1])
            for c in range(refmodel.n_components):
                stats['obs*obs[t-1].T'][c] += \
                        posteriors[t, c] * obsobsTminus1

        for t in range(1, len(seq)):
            obsobsT = np.outer(seq[t], seq[t])
            for c in range(refmodel.n_components):
                stats['obs[1:]*obs[1:].T'][c] += \
                        posteriors[t, c] * obsobsT

        for t in range(len(seq)-1):
            obsobsT = np.outer(seq[t], seq[t])
            for c in range(refmodel.n_components):
                stats['obs[:-1]*obs[:-1].T'][c] += \
                        posteriors[t, c] * obsobsT

    return curr_logprob, stats

def test_sufficient_statistics():
    # test all of the sufficient statistics against sklearn and pure python

    model = MetastableSwitchingLDS(n_states=N_STATES, n_features=refmodel.n_features)
    model._impl._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_

    logprob, stats = model._impl.do_estep()
    rlogprob, rstats = _sklearn_estep()

    yield lambda: np.testing.assert_array_almost_equal(stats['post'], rstats['post'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'], rstats['post[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'], rstats['post[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'], rstats['obs'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'], rstats['obs[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'], rstats['obs[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'], rstats['obs*obs.T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['trans'], rstats['trans'], decimal=3)

def test_gaussian_loglikelihood_full():
    _mslds.test_gaussian_loglikelihood_full()
