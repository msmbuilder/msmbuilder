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
from mslds_examples import PlusminModel, MullerModel, MullerForce
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
    return n_seq, mus, K, x_dim, T, s, xs, Ss

def test_plusmin():
    num_hotstart = 3
    num_iters = 6
    n_seq = 1
    T = 2000
    plusmin = PlusminModel()
    obs_sequences, hidden_sequences = plusmin.generate_dataset(n_seq, T)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    l = MetastableSwitchingLDS(plusmin.K, plusmin.x_dim,
            n_hotstart=num_hotstart, n_em_iter=num_iters)
    l.fit(obs_sequences)
    mslds_score = l.score(obs_sequences)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)

    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
    g.fit(obs_sequences)
    hmm_score = g.score(obs_sequences)
    print("HMM Log-Likelihood = %f" %  hmm_score)
    sim_xs, sim_Ss = l.sample(T, init_state=0, init_obs=plusmin.mus[0])
    sim_xs = np.reshape(sim_xs, (n_seq, T, plusmin.x_dim))

    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), obs_sequences[0], label="Observations")
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
    plt.legend()
    plt.show()

def test_muller_potential():
    muller = MullerModel()
    n_seq = 1
    num_trajs = 1
    sim_T = 1000
    T = 2500
    num_hotstart = 5
    num_iters = 10
    max_iters = 20
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    obs_sequences, trajectory, start = \
            muller.generate_dataset(n_seq, num_trajs, T)
    # Learn the MetastableSwitchingLDS
    l = MetastableSwitchingLDS(muller.K, muller.x_dim,
            n_hotstart=num_hotstart, n_em_iter=num_iters,
            max_iters=max_iters)
    l.fit(obs_sequences)
    mslds_score = l.score(obs_sequences)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)
    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(muller.K, muller.x_dim)
    g.fit(obs_sequences)
    hmm_score = g.score(obs_sequences)
    print("HMM Log-Likelihood = %f" %  hmm_score)

    # Clear Display
    plt.cla()
    plt.plot(trajectory[start:, 0], trajectory[start:, 1], color='k')
    plt.scatter(l.means_[:, 0], l.means_[:, 1], color='r', zorder=10)
    plt.scatter(obs_sequences[0, :, 0], obs_sequences[0,:, 1],
            edgecolor='none', facecolor='k', zorder=1)
    Delta = 0.5
    minx = min(obs_sequences[0, :, 0])
    maxx = max(obs_sequences[0, :, 0])
    miny = min(obs_sequences[0, :, 1])
    maxy = max(obs_sequences[0, :, 1])
    sim_xs, sim_Ss = l.sample(sim_T, init_state=0, init_obs=l.means_[0])

    minx = min(min(sim_xs[:, 0]), minx) - Delta
    maxx = max(max(sim_xs[:, 0]), maxx) + Delta
    miny = min(min(sim_xs[:, 1]), miny) - Delta
    maxy = max(max(sim_xs[:, 1]), maxy) + Delta
    plt.scatter(sim_xs[:, 0], sim_xs[:, 1], edgecolor='none',
               zorder=5, facecolor='g')
    plt.plot(sim_xs[:, 0], sim_xs[:, 1], zorder=5, color='g')


    MullerForce.plot(ax=plt.gca(), minx=minx, maxx=maxx, miny=miny, maxy=maxy)
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
