"""
Then, afterwards, the A, b and Q are estimated. So, we can do a lot of
testing by comparing to a reference gaussian HMM implementation
"""
import string
import numpy as np
import warnings
from sklearn.hmm import GaussianHMM
from sklearn.utils.extmath import logsumexp
from mixtape.mslds import MetastableSwitchingLDS
from mixtape import _mslds
from mixtape.ghmm import GaussianFusionHMM
import matplotlib.pyplot as plt

N_STATES = 2
data = [np.random.randn(100, 3), np.random.randn(100, 3)]
refmodel = GaussianHMM(n_components=N_STATES, covariance_type='full').fit(data)

def test_sample():
    assert True == False

def test_predict():
    assert True == False

def test_fit():
    assert True == False

def test_means_update():
    assert True == False

def test_transmat_update():
    assert True == False

def test_A_update():
    assert True == False

def test_Q_update():
    assert True == False

def test_b_update():
    assert True == False

def gen_plusmin_model():
    """The switching system has the following one-dimensional dynamics:
        x_{t+1}^1 = x_t + \epsilon_1
        x_{t+1}^2 = -x_t + \epsilon_2
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    n_seq = 1
    T = 2000
    x_dim = 1
    K = 2
    As = np.reshape(np.array([[0.6], [0.6]]), (K, x_dim, x_dim))
    bs = np.reshape(np.array([[0.4], [-0.4]]), (K, x_dim))
    Qs = np.reshape(np.array([[0.01], [0.01]]), (K, x_dim, x_dim))
    Z = np.reshape(np.array([[0.995, 0.005], [0.005, 0.995]]), (K, K))

    pi = np.reshape(np.array([0.99, 0.01]), (K,))
    mus = np.reshape(np.array([[1], [-1]]), (K, x_dim))
    Sigmas = np.reshape(np.array([[0.01], [0.01]]), (K, x_dim, x_dim))
    s = MetastableSwitchingLDS(K, x_dim)
    s.As_ = As
    s.bs_ = bs
    s.Qs_ = Qs
    s.transmat_ = Z
    s.populations_ = pi
    s.means_ = mus
    s.covars_ = Sigmas
    xs, Ss = s.sample(T)
    xs = [xs]
    return n_seq, mus, K, x_dim, T, s, xs, Ss


def test_plusmin():
    num_hotstart = 3
    num_iters = 6
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    n_seq, mus, K, x_dim, T, s, xs, Ss = gen_plusmin_model()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    l = MetastableSwitchingLDS(K, x_dim, n_hotstart=num_hotstart,
            n_em_iter=num_iters)
    l.fit(xs)
    mslds_score = l.score(xs)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)

    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(K, x_dim)
    g.fit(xs)
    hmm_score = g.score(xs)
    print("HMM Log-Likelihood = %f" %  hmm_score)
    sim_xs, sim_Ss = l.sample(T, init_state=0, init_obs=mus[0])
    sim_xs = np.reshape(sim_xs, (n_seq, T, x_dim))

    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), xs[0], label="Observations")
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
    plt.legend()
    plt.show()

def reference_estep():
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

    yield lambda: np.testing.assert_array_almost_equal(stats['post'],
            rstats['post'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'],
            rstats['post[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'],
            rstats['post[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'],
            rstats['obs'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'],
            rstats['obs[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'],
            rstats['obs[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'],
            rstats['obs*obs.T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['trans'], rstats['trans'], decimal=3)

def test_gaussian_loglikelihood_full():
    _mslds.test_gaussian_loglikelihood_full()
