from __future__ import print_function, division
import numpy as np
import warnings
import mdtraj as md
from mixtape.datasets import load_doublewell
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mixtape.mslds import MetastableSwitchingLDS
from mixtape.ghmm import GaussianFusionHMM
import matplotlib.pyplot as plt
from mixtape.datasets.alanine_dipeptide import fetch_alanine_dipeptide
from mixtape.datasets.alanine_dipeptide import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_ALANINE
from mixtape.datasets.base import get_data_home
from os.path import join
from sklearn.mixture.gmm import log_multivariate_normal_density
from nose.plugins.attrib import attr


@attr('plots')
def test_plusmin():
    # Set constants
    n_hotstart = 3
    n_em_iter = 3
    n_experiments = 1
    n_seq = 1
    T = 2000
    gamma = 512.

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = plusmin.x_dim
    n_components = plusmin.K

    # Train MSLDS
    mslds_scores = []
    l = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=n_hotstart, n_em_iter=n_em_iter,
            n_experiments=n_experiments)
    l.fit(data, gamma=gamma)
    mslds_score = l.score(data)
    print("gamma = %f" % gamma)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)
    print()

    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
    g.fit(data)
    hmm_score = g.score(data)
    print("HMM Log-Likelihood = %f" %  hmm_score)
    print()

    # Plot sample from MSLDS
    sim_xs, sim_Ss = l.sample(T, init_state=0, init_obs=plusmin.mus[0])
    sim_xs = np.reshape(sim_xs, (n_seq, T, plusmin.x_dim))
    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), data[0], label="Observations")
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
    plt.legend()
    plt.show()


@attr('plots')
def test_muller_potential():
    import pdb, traceback, sys
    try:
        # Set constants
        n_hotstart = 3
        n_em_iter = 3
        n_experiments = 1
        n_seq = 1
        num_trajs = 1
        T = 2500
        sim_T = 2500
        gamma = 200. 

        # Generate data
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        muller = MullerModel()
        data, trajectory, start = \
                muller.generate_dataset(n_seq, num_trajs, T)
        n_features = muller.x_dim
        n_components = muller.K

        # Train MSLDS
        model = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=n_hotstart, n_em_iter=n_em_iter,
            n_experiments=n_experiments)
        model.fit(data, gamma=gamma)
        mslds_score = model.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)

        # Clear Display
        plt.cla()
        plt.plot(trajectory[start:, 0], trajectory[start:, 1], color='k')
        plt.scatter(model.means_[:, 0], model.means_[:, 1], 
                    color='r', zorder=10)
        plt.scatter(data[0][:, 0], data[0][:, 1],
                edgecolor='none', facecolor='k', zorder=1)
        Delta = 0.5
        minx = min(data[0][:, 0])
        maxx = max(data[0][:, 0])
        miny = min(data[0][:, 1])
        maxy = max(data[0][:, 1])
        sim_xs, sim_Ss = model.sample(sim_T, init_state=0,
                init_obs=model.means_[0])

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


@attr('plots')
def test_doublewell():
    import pdb, traceback, sys
    try:
        n_components = 2
        n_features = 1
        n_em_iter = 1
        n_experiments = 1
        tol=1e-1

        data = load_doublewell(random_state=0)['trajectories']
        T = len(data[0])

        # Fit MSLDS model 
        model = MetastableSwitchingLDS(n_components, n_features,
            n_experiments=n_experiments, n_em_iter=n_em_iter)
        model.fit(data, gamma=.1, tol=tol)
        mslds_score = model.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print()

        # Plot sample from MSLDS
        sim_xs, sim_Ss = model.sample(T, init_state=0)
        plt.close('all')
        plt.figure(1)
        plt.plot(range(T), data[0], label="Observations")
        plt.plot(range(T), sim_xs, label='Sampled Observations')
        plt.legend()
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

@attr('plots')
def test_alanine_dipeptide():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    try:
        b = fetch_alanine_dipeptide()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        sim_T = 1000
        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_ALANINE)
        top = md.load(join(data_dir, 'ala2.pdb'))
        n_components = 2
        # Superpose m
        data = []
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 1
        tol = 1e-1
        model = MetastableSwitchingLDS(n_components, 
            n_features, n_experiments=n_experiments, 
            n_em_iter=n_em_iter) 
        model.fit(data, gamma=.1, tol=tol, verbose=True)
        mslds_score = model.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print()

        # Generate a trajectory from learned model.
        sample_traj, hidden_states = model.sample(sim_T)
        states = []
        for k in range(n_components):
            states.append([])

        # Presort the data into the metastable wells
        for k in range(n_components):
            for i in range(len(trajs)):
                traj = trajs[i]
                Z = traj.xyz
                Z = np.reshape(Z, (len(Z), n_features), order='F')
                logprob = log_multivariate_normal_density(Z,
                    np.array(model.means_),
                    np.array(model.covars_), covariance_type='full')
                assignments = np.argmax(logprob, axis=1)
                #probs = np.max(logprob, axis=1)
                # pick structures that have highest log probability in state
                s = traj[assignments == k]
                states[k].append(s)

        # Pick frame from original trajectories closest to current sample
        gen_traj = None
        for t in range(sim_T):
            h = hidden_states[t]
            best_logprob = -np.inf
            best_frame = None
            for i in range(len(trajs)):
                if t > 0:
                    states[h][i].superpose(gen_traj, t-1)
                Z = states[h][i].xyz
                Z = np.reshape(Z, (len(Z), n_features), order='F')
                mean = sample_traj[t]
                logprobs = log_multivariate_normal_density(Z,
                    mean, model.Qs_[h], covariance_type='full')
                ind = np.argmax(logprobs, axis=0)
                logprob = logprobs[ind]
                if logprob > best_log_prob:
                    logprob = best_logprob
                    best_frame = states[h][i][ind]
            if t == 0:
                gen_traj = best_frame
            else:
                gen_traj = gen_traj.join(frame)
        gen_traj.save('%s.xtc' % self.out)
        gen_traj[0].save('%s.xtc.pdb' % self.out)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
