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
from mixtape.datasets.met_enkephalin import fetch_met_enkephalin
from mixtape.datasets.met_enkephalin import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_MET
from mixtape.datasets.src_kinase import fetch_src_kinase
from mixtape.datasets.src_kinase import src_kinase_atom_indices
from mixtape.datasets.src_kinase import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_SRC
from mixtape.datasets.src_kinase import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_NANO
from mixtape.datasets.base import get_data_home
from os.path import join
from mixtape.utils import save_mslds_to_json_dict
from mixtape.utils import gen_trajectory, project_trajectory


def test_alanine():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    LEARN = True
    try:
        b = fetch_alanine_dipeptide()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        n_components = 2
        atom_indices = range(n_atoms)
        sim_T = 100
        gamma = .05
        out = "alanine_test"

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_ALANINE)
        top = md.load(join(data_dir, 'ala2.pdb'))
        # Superpose m
        data = []
        # For debugging
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 3
        tol = 2e-1
        search_tol = 1
        if LEARN:
            model = MetastableSwitchingLDS(n_components, 
                n_features, n_experiments=n_experiments, 
                n_em_iter=n_em_iter) 
            model.fit(data, gamma=gamma, tol=tol, verbose=False,
                        search_tol=search_tol, stable=False)
            mslds_score = model.score(data)
            print("MSLDS Log-Likelihood = %f" %  mslds_score)

            # Save the learned model
            save_mslds_to_json_dict(model, 'alanine_no_stability.json')
            # Generate a trajectory from learned model.
            sample_traj, hidden_states = model.sample(sim_T)
        else:
            sample_traj = np.random.rand(sim_T, n_features)
            hidden_states = np.random.randint(n_components, size=(sim_T,))

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        gen_trajectory(sample_traj, hidden_states, n_components, 
                        n_features, trajs, out, g, sim_T, atom_indices)
        import pickle
        pickle.dump(sample_traj, open("sample_traj.p", "w"))
        pickle.dump(hidden_states, open("hidden_states.p", "w"))

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_met_enk():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    LEARN = True
    try:
        print "About to fetch trajectories"
        b = fetch_met_enkephalin()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        atom_indices = range(n_atoms)
        n_features = n_atoms * 3
        n_components = 2
        gamma = .1
        sim_T = 100
        out = "met_enk_test"

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_MET)
        top = md.load(join(data_dir, '1plx.pdb'))
        # Superpose m
        data = []
        trajs = trajs
        for traj in trajs:
            "Superposing Trajectory"
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 3
        tol = 2e-1
        search_tol = 1.
        if LEARN:
            model = MetastableSwitchingLDS(n_components, 
                n_features, n_experiments=n_experiments, 
                n_em_iter=n_em_iter) 
            model.fit(data, gamma=gamma, tol=tol, verbose=False,
                    search_tol=search_tol, stable=False)
            mslds_score = model.score(data)
            print("MSLDS Log-Likelihood = %f" %  mslds_score)

            # Save the learned model
            save_mslds_to_json_dict(model, 'met_enk_no_stability.json')
            # Generate a trajectory from learned model.
            sample_traj, hidden_states = model.sample(sim_T)
        else:
            sample_traj = np.random.rand(sim_T, n_features)
            hidden_states = np.random.randint(n_components, size=(sim_T,))

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        gen_trajectory(sample_traj, hidden_states, n_components, 
                        n_features, trajs, out, g, sim_T, atom_indices)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
