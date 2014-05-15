import numpy as np
import warnings
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mixtape.mslds import MetastableSwitchingLDS
from sklearn.hmm import GaussianHMM

def test_plusmin_mstep():
    # Set constants
    num_hotstart = 3
    n_seq = 1
    T = 2000

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = plusmin.x_dim
    n_components = plusmin.K

    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Fit initial MSLDS model from reference model
    model = MetastableSwitchingLDS(n_components, n_features,
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

    # Do one mstep
    params = 'aqb' # With param t is broken!
    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model._impl.do_estep(iteration)
    model._do_mstep(stats, params, iteration)

    #import pdb
    #pdb.set_trace()
    for i in range(n_components):
        yield lambda: np.testing.assert_array_almost_equal(
                model.Qs_[i], refmodel.covars_[i], decimal=2)
        yield lambda: np.testing.assert_array_almost_equal(
                model.bs_[i], model.means_[i], decimal=2)
        yield lambda: np.testing.assert_array_almost_equal(
                model.As_[i], np.zeros((n_features, n_features)),
                decimal=1)

def test_plusmin_scores():
    l = MetastableSwitchingLDS(plusmin.K, plusmin.x_dim,
           n_hotstart=0)
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
    pass

def test_muller_potential_mstep():
    # Set constants
    n_seq = 1
    num_trajs = 1
    T = 2500
    num_hotstart = 0

    # Generate data
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    muller = MullerModel()
    data, trajectory, start = \
            muller.generate_dataset(n_seq, num_trajs, T)
    n_features = muller.x_dim
    n_components = muller.K

    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    model = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=num_hotstart)
    model._impl._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    As = []
    for i in range(n_components):
        As.append(np.zeros((n_features, n_features)))
    model.As_ = As
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    # Do one mstep
    params = 'aqb' # With param t is broken!
    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model._impl.do_estep(iteration)
    model._do_mstep(stats, params, iteration)

    # The current implementation's behavior is pretty broken here....
    # So this test should auto-fail until things are fixed.
    assert True == False

def test_muller_potential_score():
    sim_T = 1000
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
    pass

