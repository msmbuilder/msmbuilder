"""
Then, afterwards, the A, b and Q are estimated. So, we can do a lot of
testing by comparing to a reference gaussian HMM implementation
"""
import string
import numpy as np
import warnings
import os
from sklearn.hmm import GaussianHMM
from sklearn.utils.extmath import logsumexp
from mixtape.mslds import MetastableSwitchingLDS
from mixtape import _mslds
from mixtape.ghmm import GaussianFusionHMM
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mslds_examples import AlanineDipeptideModel
import matplotlib.pyplot as plt


def test_plusmin_estep():
    # Set constants
    num_hotstart = 3
    num_iters = 6
    n_seq = 1
    T = 2000

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = 1
    n_components = plusmin.K

    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    model = MetastableSwitchingLDS(plusmin.K, plusmin.x_dim,
                                n_hotstart=0)
    model._impl._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    model.As_ = [np.zeros((n_features, n_features)),
                    np.zeros((n_features, n_features))]
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model._impl.do_estep(iteration)
    rlogprob, rstats = reference_estep(refmodel, data)

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


def test_muller_potential_estep():
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


    MullerForce.plot(ax=plt.gca(), minx=minx, maxx=maxx,
            miny=miny, maxy=maxy)
    plt.show()

def test_alanine_estep():
    K = 2
    T = 1000
    num_trajs = 1
    traj_filename = 'ala2.h5'
    a = AlanineDipeptideModel()
    if not os.path.exists(traj_filename):
        a.generate_dataset(traj_filename, T)
    xs, x_dim = a.load_dataset(traj_filename)
    sequences = [xs]
    import pdb
    pdb.set_trace()

    # Fit Metastable Switcher
    num_hotstart = 1
    num_iters = 2
    l = MetastableSwitchingLDS(K, x_dim, n_hotstart=num_hotstart,
            n_em_iter=num_iters)
    l.fit(sequences)
    mslds_score = l.score(sequences)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)

    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(K, x_dim)
    g.fit(sequences)
    hmm_score = g.score(sequences)
    print("HMM Log-Likelihood = %f" %  hmm_score)

    sim_xs,sim_Ss,sim_ys = l.simulate(sim_T,s_init=0, x_init=means[0],
      y_init=means[0])
    gen_movie(sim_ys, topology, 'alanine', sim_T, N_atoms, dim)
    gen_movie(out_x_tTs[NUM_SCHED-1,NUM_ITERS-1],
      topology, 'alanine2', T, N_atoms, dim)

def reference_estep(refmodel, data):
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

def test_sufficient_statistics_basic():
    """
    Sanity test MSLDS sufficient statistic gathering by setting
    dynamics model to 0 and testing that E-step matches that of
    HMM
    """
    # Generate reference data
    n_states = 2
    n_features = 3
    data = [np.random.randn(100, n_features),
            np.random.randn(100, n_features)]
    refmodel = GaussianHMM(n_components=n_states,
                        covariance_type='full').fit(data)

    # test all of the sufficient statistics against sklearn and pure python

    model = MetastableSwitchingLDS(n_states=n_states,
            n_features=n_features, n_hotstart=0)
    model._impl._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    # Is there a more elegant way to do this?
    model.As_ = [np.zeros((n_features, n_features)),
                    np.zeros((n_features, n_features))]
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model._impl.do_estep(iteration)
    rlogprob, rstats = reference_estep(refmodel, data)

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
