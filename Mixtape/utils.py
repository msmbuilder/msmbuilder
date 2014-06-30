"""Utility functions"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import json
import numpy as np
from sklearn.utils import check_random_state
from sklearn.externals.joblib import load, dump
from numpy.linalg import norm
import pickle
from sklearn.mixture.gmm import log_multivariate_normal_density
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def verbosedump(value, fn, compress=1):
    """verbose wrapper around joblib.dump"""
    print('Saving "%s"... (%s)' % (fn, type(value)))
    dump(value, fn, compress=compress)

def verboseload(fn):
    """verbose wrapper around joblib.load"""
    print('loading "%s"...' % fn)
    return load(fn)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass


def categorical(pvals, size=None, random_state=None):
    """Return random integer from a categorical distribution

    Parameters
    ----------
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1.
    size : int or tuple of ints, optional
        Defines the shape of the returned array of random integers. If None
        (the default), returns a single float.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    """
    cumsum = np.cumsum(pvals)
    if size is None:
        size = (1,)
        axis = 0
    elif isinstance(size, tuple):
        size = size + (1,)
        axis = len(size) - 1
    else:
        raise TypeError('size must be an int or tuple of ints')

    random_state = check_random_state(random_state)
    return np.sum(cumsum < random_state.random_sample(size), axis=axis)


##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V


def print_solve_test_case(name, matrices, dim, test_file):
    disp = "%s_solve failed. " % name
    disp += "Autogenerating %s test case" % name
    disp = (bcolors.FAIL + disp + bcolors.ENDC)
    test_num = np.random.randint(1000)
    print(disp)
    with open(test_file, 'a') as f:
        disp  = ""
        np.set_printoptions(threshold=np.nan)
        disp += "\ndef %s_test():\n" % name
        disp += "\t#Auto-generated test case from failing run of\n"
        disp += "\t#%s-solve:\n"%name
        disp += "\timport numpy as np\n"
        disp += "\timport pickle\n"
        disp += "\timport time\n"
        disp += "\tfrom mixtape.mslds_solvers.A_problems import A_problem\n"
        disp += "\tfrom mixtape.mslds_solvers.Q_problems import Q_problem\n"
        disp += "\tn_features = %d\n"%dim
        disp += "\t%s_prob = %s_problem(n_features)\n"%(name.lower(), name)
        arg_string = ""
        for mat, mat_name in matrices:
            pickle.dump(mat, open("%s_%s_test%d.p" 
                                    % (mat_name, name, test_num), "w"))
            disp += ('\t%s = pickle.load(open("%s_%s_test%d.p", "r"))\n'
                            % (mat_name, mat_name, name, test_num))
            arg_string += mat_name + ", "
        disp += "\t%s_prob.solve(%s\n" % (name.lower(), arg_string)
        disp += "\t\tdisp=True, debug=False, verbose=True)\n"
        f.write(disp)
    np.set_printoptions(threshold=1000)

def sample_hmm(n_samples, n_features, hidden_states, means, covars):
    obs = np.zeros((n_samples, n_features))
    for t in range(n_samples):
        s = hidden_states[t]
        val = multivariate_normal(means[s], covars[s])
        obs[t] = val
    return obs

def project_trajectory(sample_traj, means, atom_index_per_mean):
    sim_T = len(sample_traj)
    red_dim = len(means)
    projected = np.zeros((sim_T, red_dim))
    for t in range(sim_T):
        for i in range(red_dim):
            ind = atom_index_per_mean[i]
            projected[t,i] = sample_traj[t][ind] - means[i][ind]
    return projected

def project_trajectory_simple(sample_traj, means):
    sim_T = len(sample_traj)
    red_dim = len(means)
    projected = np.zeros((sim_T, red_dim))
    for t in range(sim_T):
        for i in range(red_dim):
            projected[t,i] = np.linalg.norm(sample_traj[t] - means[i])
    return projected

#def plot_mult_coords(trajs, means, atom_indices_per_mean, colors):
#    for traj, mean, a

def plot_coords(sample_traj, means, atom_indices_per_mean, scatter=False,
                simple=True,
                xlabel="Order Parameter 1", ylabel="Order Parameter 2"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if simple:
        projected = project_trajectory_simple(sample_traj, means)
    else:
        projected = project_trajectory(sample_traj, means,
                        atom_indices_per_mean)
    T = np.shape(projected)[0]
    if scatter:
        ax.scatter(projected[:, 0], projected[:, 1],
            color=np.arange(T)/T)
    else:
        ax.plot(projected[:, 0], projected[:, 1], 
            color=np.arange(T)/T)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return projected


def gen_trajectory(sample_traj, hidden_states, n_components, n_features,
        trajs, out, g, sim_T, atom_indices):
    states = []
    for k in range(n_components):
        states.append([])

    # Presort the data into the metastable wells
    for k in range(n_components):
        print("Presorting component %d" % k)
        for i in range(len(trajs)):
            print("\tIn trajectory %d" % i)
            traj = trajs[i]
            Z = traj.xyz[:, atom_indices]
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            logprob = log_multivariate_normal_density(Z,
                np.array(g.means_), np.array(g.vars_), 
                covariance_type='diag')
            assignments = np.argmax(logprob, axis=1)
            s = traj[assignments == k]
            states[k].append(s)

    # Pick frame from original trajectories closest to current sample
    gen_traj = None
    for t in range(sim_T):
        print("t = %d" % t)
        h = hidden_states[t]
        best_dist = np.inf
        best_frame = None
        for i in range(len(trajs)):
            if t > 0:
                states[h][i].superpose(gen_traj, t-1)
            Z = states[h][i].xyz[:, atom_indices]
            if len(Z) == 0:
                continue
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            cur_sample = sample_traj[t]
            cur_sample = np.tile(cur_sample, (len(Z), 1))
            diffs = Z - cur_sample
            dists = np.sum(diffs**2, axis=1)
            ind = np.argmin(dists)
            dist = dists[ind]
            if dist < best_dist:
                best_dist = dist 
                best_frame = states[h][i][ind]
        if t == 0:
            gen_traj = best_frame
        else:
            gen_traj = gen_traj.join(best_frame)
    gen_traj.save('%s.xtc' % out)
    gen_traj[0].save('%s.xtc.pdb' % out)


def save_mslds_to_json_dict(model, out):
    with open(out, 'w') as outfile:
        result = {
            'model': 'MetastableSwitchingLinearDynamicalSystem',
            'n_states': model.n_states,
            'n_features': model.n_features,
            'transmat': model.transmat_.tolist(),
            'means': model.means_.tolist(),
            'covars': model.covars_.tolist(),
            'As': model.As_.tolist(),
            'bs': model.bs_.tolist(),
            'Qs': model.Qs_.tolist(),
        }

        if not np.all(np.isfinite(model.transmat_)):
            raise ValueError('Nonfinite numbers in transmat!')

        json.dump(result, outfile)
        outfile.write('\n')

def load_mslds_from_json_dict(out):
    # Place import here to avoid weird circular dependencies
    from mixtape.mslds import MetastableSwitchingLDS
    with open(out, 'r') as outfile:
        model_dict = json.load(outfile)
        # Check that the num of states and features agrees
        n_features = float(model_dict['n_features'])
        n_components = float(model_dict['n_states'])
        # read array values from the json dictionary
        Qs = []
        for Q in model_dict['Qs']:
            Qs.append(np.array(Q))
        As = []
        for A in model_dict['As']:
            As.append(np.array(A))
        bs = []
        for b in model_dict['bs']:
            bs.append(np.array(b))
        means = []
        for mean in model_dict['means']:
            means.append(np.array(mean))
        covars = []
        for covar in model_dict['covars']:
            covars.append(np.array(covar))
        # Transmat
        transmat = np.array(model_dict['transmat'])
        # Create the MSLDS model
        model = MetastableSwitchingLDS(n_components, n_features) 
        model.Qs_ = Qs
        model.As_ = As
        model.bs_ = bs
        model.means_ = means
        model.covars_ = covars
        model.transmat_ = transmat
        return model

##########################################################################
# END of MSLDS Utils (experimental)
##########################################################################

def map_drawn_samples(selected_pairs_by_state, trajectories):
    """Lookup trajectory frames using pairs of (trajectory, frame) indices.

    Parameters
    ----------
    selected_pairs_by_state : np.ndarray, dtype=int, shape=(n_states, n_samples, 2)
        selected_pairs_by_state[state, sample] gives the (trajectory, frame)
        index associated with a particular sample from that state.
    trajectories : list(md.Trajectory)
        The trajectories assocated with sequences,
        which will be used to extract coordinates of the state centers
        from the raw trajectory data

    Returns
    -------
    frames_by_state : mdtraj.Trajectory, optional
        If `trajectories` are provided, this output will be a list
        of trajectories such that frames_by_state[state] is a trajectory
        drawn from `state` of length `n_samples`
    
    Examples
    --------
    >>> selected_pairs_by_state = hmm.draw_samples(sequences, 3)
    >>> samples = map_drawn_samples(selected_pairs_by_state, trajectories)
    
    Notes
    -----
    YOU are responsible for ensuring that selected_pairs_by_state and 
    trajectories correspond to the same dataset!
    
    See Also
    --------
    utils.map_drawn_samples : Extract conformations from MD trajectories by index.
    ghmm.GaussianFusionHMM.draw_samples : Draw samples from GHMM    
    ghmm.GaussianFusionHMM.draw_centroids : Draw centroids from GHMM    
    """

    frames_by_state = []

    for state, pairs in enumerate(selected_pairs_by_state):
        frames = [trajectories[trj][frame] for trj, frame in pairs]
        state_trj = np.sum(frames)  # No idea why numpy is necessary, but it is
        frames_by_state.append(state_trj)
    
    return frames_by_state

