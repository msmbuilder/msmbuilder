'''Generate a trajectory from a MSLDS Model'''
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
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

import os, sys, pdb
import glob
import numpy as np
import pandas as pd
import mdtraj as md
from sklearn.mixture.gmm import log_multivariate_normal_density

from mixtape.utils import iterobjects
from mixtape.cmdline import argument_group
from mixtape.commands.sample import SampleGHMM
import mixtape.featurizer
from mixtape.cmdline import Command, argument_group, MultipleIntAction
from mixtape.commands.mixins import MDTrajInputMixin
from mixtape.mslds import *
import traceback, sys, code, pdb

__all__ = ['PullMeansGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SampleMSLDS(Command, MDTrajInputMixin):
    name = 'sample-mslds'
    description = '''Sample a trajectory from a MSLDS model.'''

    group = argument_group('I/O Arguments')
    group.add_argument('-i', '--filename', required=True,
        metavar='JSONLINES_FILE', help=
        '''Path to the jsonlines output file containg the MSLDS''')
    group.add_argument('--featurizer', type=str, required=True,
        help='''Path to saved featurizer object''')
    group.add_argument('--stride', type=int, default=1, help='''
        Load up only every stride-th frame from the trajectories, to reduce
        memory usage''')
    group.add_argument('--n-states', type=int, required=True,
        help='''Number of states in the model to select from''')
    group.add_argument('--n-samples', type=int, required=True,
        help='''Length of trajectory to sample''')
    group.add_argument('-o', '--out', metavar='OUTPUT_XTC_FILE',
        default='traj',
        help=('''File to which to save the output, ''' +
                '''in xtc format. default="traj'''))
    group.add_argument('--use-pdb', action='store_true',
		help= ('''Launch python debugger PDB on exception. ''' +
              '''Useful for debugging.'''))

    def __init__(self, args):
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        matches = [o for o in iterobjects(args.filename)
                   if o['n_states'] == args.n_states]
        if len(matches) == 0:
            self.error('No model with n_states=%d in %s.'
               % (args.n_states, args.filename))

        self.args = args
        self.model_dict = matches[0]
        self.out = args.out
        self.topology = md.load(args.top)
        self.filenames = glob.glob(
            os.path.join(os.path.expanduser(args.dir), '*.%s' % args.ext))
        self.featurizer = mixtape.featurizer.load(args.featurizer)
        self.stride = stride

        if len(self.filenames) == 0:
            self.error('No files matched.')

    def start(self):
        args = self.args
        if self.args.use_pdb:
            try:
                self._start()
            except:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
        else:
            self._start()

    def _start(self):
        print("model")
        print(self.model_dict)
        n_features = float(self.model_dict['n_features'])
        n_states = float(self.model_dict['n_states'])
        self.model = MetastableSwitchingLDS(n_states, n_features)
        self.model.load_from_json_dict(self.model_dict)
        obs, hidden_states = self.model.sample(self.args.n_samples)
        (n_samples, n_features) = np.shape(obs)

        features, ii, ff = mixtape.featurizer.featurize_all(
            self.filenames, self.featurizer, self.topology, self.stride)
        file_trajectories = []

        states = []
        state_indices = []
        state_files = []
        logprob = log_multivariate_normal_density(
            features, np.array(self.model.means_),
            np.array(self.model.covars_), covariance_type='full')
        assignments = np.argmax(logprob, axis=1)
        probs = np.max(logprob, axis=1)
        # Presort the data into the metastable wells
        # i.e.: separate the original trajectories into k
        # buckets corresponding to the metastable wells
        for k in range(int(self.model.n_states)):
            # pick the structures that have the highest log
            # probability in the state
            s = features[assignments == k]
            ind = ii[assignments==k]
            f = ff[assignments==k]
            states.append(s)
            state_indices.append(ind)
            state_files.append(f)

        # Loop over the generated feature space trajectory.
        # At time t, pick the frame from the original trajectory
        # closest to the current sample in feature space. To save
        # a bit of computation, just search in the bucket corresponding
        # to the current metastable well (i.e., the current hidden state).
        traj = None
        for t in range(n_samples):
            featurized_frame = obs[t]
            h = hidden_states[t]
            logprob = log_multivariate_normal_density(
                states[h], featurized_frame[np.newaxis],
                self.model.Qs_[h][np.newaxis],
                covariance_type='full')
            best_frame_pos = np.argmax(logprob, axis=0)[0]
            best_file = state_files[h][best_frame_pos]
            best_ind = state_indices[h][best_frame_pos]
            frame = md.load_frame(best_file, best_ind, self.topology)
            if t == 0:
                traj = frame
            else:
                frame.superpose(traj, t-1)
                traj = traj.join(frame)
        traj.save('%s.xtc' % self.out)
        traj[0].save('%s.xtc.pdb' % self.out)
        #pdb.set_trace()
