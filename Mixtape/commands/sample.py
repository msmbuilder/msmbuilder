from __future__ import print_function
import os
import sys
import glob
import numpy as np
import mdtraj as md
import pandas as pd

from mixtape.utils import iterobjects, load_superpose_timeseries
from mixtape.discrete_mvn import discrete_approx_mvn
from mixtape.cmdline import Command, argument, argument_group

class SampleGHMM(Command):
    name = 'sample-ghmm'
    description = '''Draw iid samples from each state in an Gaussian HMM.

    The output is a small CSV file with 3 columns: 'filename', 'index',
    and 'state'. Each row gives the path to a trajectory file, the index
    of a single frame therein, and the state it was drawn from.

    The sampling strategy is as follows: for each state represented by a
    Gaussian distribution, we create a discrete distribution over the
    featurized frames in the specified trajectory files such that the
    discrete distribution has the same mean and variance as the state Gaussian
    distribution and minimizes the K-L divergence from the discrete distribution
    to the continuous Gaussian it's trying to model. Then, we sample from that
    discrete distribution and return the corresponding frames in a CSV file.

    The reason for this complexity is that the Gaussian distributions for
    each state are continuous distributions over the featurized space. To
    visualize the structures corresponding to each state, however, we would
    need to sample from this distribution and then "invert" the featurization,
    to reconstruct the cartesian coordinates for our samples. Alternatively,
    we can draw from a discrete distribution over our available structures;
    but this introduces the question of what discrete distribution "optimally"
    represents the continuous (Gaussian) distribution of interest.


    [Reference]: Tanaka, Ken'ichiro, and Alexis Akira Toda. "Discrete
    approximations of continuous distributions by maximum entropy."
    Economics Letters 118.3 (2013): 445-450.
    '''

    fn = argument('--filename', required=True, help='''Path to the jsonlines output
        file containg the HMMs''')
    ns = argument('--n-states', type=int, required=True, help='''Number of states in
        the model to select from''')
    nps = argument('--n-per-state', type=int, default=3, help='''Number of structures
        to pull from each state''')
    lt = argument('--lag-time', type=int, required=True, help='''Training lag
        time of the model to select from''')
    top = argument('--top', metavar='PDB_FILE', required=True,
                   help='Topology file for the system')
    dir = argument('--dir', metavar='DIRECTORY', required=True,
                   help='Directory containing the trajectory files')
    ext = argument('--ext', metavar='EXTENSION', required=True,
                   help='File extension of the trajectory files')
    out = argument('-o', '--out', metavar='OUTPUT_CSV_FILE', required=True,
                   help='File to which to save the output, in CSV format')

    group_munge = argument_group('Munging Options')
    group_vector = group_munge.add_mutually_exclusive_group(required=True)
    group_vector.add_argument('-d', '--distance-pairs', type=str, help='''Vectorize
        the MD trajectories by extracting timeseries of the distance
        between pairs of atoms in each frame. Supply a text file where
        each row contains the space-separate indices of two atoms which
        form a pair to monitor''')
    group_vector.add_argument('-a', '--atom-indices', type=str, help='''Superpose
        each MD conformation on the coordinates in the topology file, and then use
        the distance from each atom in the reference conformation to the
        corresponding atom in each MD conformation.''')

    def __init__(self, args):
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        matches = [o for o in iterobjects(args.filename)
                   if o['n_states'] == args.n_states
                   and o['train_lag_time'] == args.lag_time]
        if len(matches) == 0:
            self.error('No model with n_states=%d, train_lag_time=%d in %s.' % (
                args.n_states, args.lag_time, args.filename))

        self.model = matches[0]
        self.out = args.out
        self.n_per_state = args.n_per_state
        self.topology = md.load(args.top)
        self.filenames = glob.glob(os.path.join(os.path.expanduser(args.dir), '*.%s' % args.ext))
        self.atom_indices = np.loadtxt(args.atom_indices, dtype=int, ndmin=1)

        if len(self.filenames) == 0:
            self.error('No files matched.')
        if args.distance_pairs is not None:
            raise NotImplementedError()

    def start(self):
        print('loading all data...')
        xx, ii, ff = load_superpose_timeseries(self.filenames, self.atom_indices, self.topology)
        print('done loading')

        data = {'filename': [], 'index': [], 'state': []}
        for k in range(self.model['n_states']):
            weights = discrete_approx_mvn(xx, self.model['means'][k], self.model['vars'][k])
            cumsum = np.cumsum(weights)
            for i in range(self.n_per_state):
                index = np.sum(cumsum < np.random.rand())
                data['filename'].append(ff[index])
                data['index'].append(ii[index])
                data['state'].append(k)

        df = pd.DataFrame(data)
        print('Saving the indices of the sampled states in CSV format to %s' % self.out)
        df.to_csv(self.out)
