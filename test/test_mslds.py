import numpy as np
import warnings
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mixtape.mslds import MetastableSwitchingLDS
from mixtape.ghmm import GaussianFusionHMM
import matplotlib.pyplot as plt

def test_plusmin():
    # Set constants
    num_hotstart = 3
    n_seq = 1
    T = 2000

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = plusmin.x_dim
    n_components = plusmin.K

    # Train MSLDS
    l = MetastableSwitchingLDS(n_components, n_features, n_hotstart=3,
            n_em_iter=1, n_experiments=1)
    l.fit(data)
    mslds_score = l.score(data)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)
    ## Fit Gaussian HMM for comparison
    #g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
    #g.fit(data)
    #hmm_score = g.score(data)
    #print("HMM Log-Likelihood = %f" %  hmm_score)
    sim_xs, sim_Ss = l.sample(T, init_state=0, init_obs=plusmin.mus[0])
    sim_xs = np.reshape(sim_xs, (n_seq, T, plusmin.x_dim))

    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), data[0], label="Observations")
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
    plt.legend()
    plt.show()
    pass

def test_muller_potential():
    import pdb, traceback, sys
    try:
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

        # Train MSLDS
        l = MetastableSwitchingLDS(n_components, n_features, n_hotstart=3,
                n_em_iter=1, n_experiments=1)
        l.fit(data)
        mslds_score = l.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)
        ## Fit Gaussian HMM for comparison
        #g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
        #g.fit(data)
        #hmm_score = g.score(data)
        #print("HMM Log-Likelihood = %f" %  hmm_score)

        # Clear Display
        plt.cla()
        plt.plot(trajectory[start:, 0], trajectory[start:, 1], color='k')
        plt.scatter(l.means_[:, 0], l.means_[:, 1], color='r', zorder=10)
        plt.scatter(data[0][:, 0], data[0][:, 1],
                edgecolor='none', facecolor='k', zorder=1)
        Delta = 0.5
        minx = min(data[0][:, 0])
        maxx = max(data[0][:, 0])
        miny = min(data[0][:, 1])
        maxy = max(data[0][:, 1])
        sim_xs, sim_Ss = l.sample(sim_T, init_state=0,
                init_obs=l.means_[0])

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
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

