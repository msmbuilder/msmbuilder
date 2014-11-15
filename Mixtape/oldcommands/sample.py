'''Sample from each state in an HMM
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
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

import os
import sys
import glob
import numpy as np
import mdtraj as md
import pandas as pd

from ..utils import iterobjects
from ..hiddenmarkovmodel.discrete_approx import discrete_approx_mvn, \
    NotSatisfiableError
from ..cmdline import FlagAction, Command, argument, argument_group
from .mixins import MDTrajInputMixin
from ..featurizer import load as feat_load, featurize_all

__all__ = ['SampleGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SampleGHMM(Command, MDTrajInputMixin):
    _concrete = True
    name = 'sample-ghmm'
    description = '''Draw iid samples from each state in a Gaussian HMM.

    The output is a small CSV file with 3 columns: 'filename', 'index',
    and 'state'. Each row gives the path to a trajectory file, the index
    of a single frame therein, and the state it was drawn from.

    The sampling strategy is as follows: for each state represented by a
    Gaussian distribution, we create a discrete distribution over the
    featurized frames in the specified trajectory files such that the
    discrete distribution has the same mean (and optionally variance) as the
    state Gaussian distribution and minimizes the K-L divergence from the
    discrete distribution to the continuous Gaussian it's trying to model. Then,
    we sample from that discrete distribution and return the corresponding
    frames in a CSV file.

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

    group = argument_group('I/O Arguments')
    group.add_argument('-i', '--filename', required=True, metavar='JSONLINES_FILE', help='''
        Path to the jsonlines output file containg the HMMs''')
    group.add_argument('--featurizer', type=str, required=True, help='''
        Path to saved featurizer object''')
    group.add_argument('--stride', type=int, default=1, help='''
        Load up only every stride-th frame from the trajectories, to reduce
        memory usage''')
    group.add_argument('--n-states', type=int, required=True, help='''Number of
        states in the model to select from''')
    group.add_argument('--n-per-state', type=int, default=3, help='''Number of
        structures to pull from each state''')
    group.add_argument('--lag-time', type=int, required=True, help='''Training lag
        time of the model to select from''')
    group.add_argument('-o', '--out', metavar='OUTPUT_CSV_FILE', default='samples.csv', help='''
        File to which to save the output, in CSV format. default="samples.csv"''')

    match_vars = argument('--match-vars', action=FlagAction, default=True, help='''
        Constrain the discrete distribution to match the variances of the
        continuous distribution. default=enabled''')

    def __init__(self, args):
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        matches = [o for o in iterobjects(args.filename)
                   if o['n_states'] == args.n_states
                   and o['train_lag_time'] == args.lag_time]
        if len(matches) == 0:
            self.error('No model with n_states=%d, train_lag_time=%d in %s.' % (
                args.n_states, args.lag_time, args.filename))

        self.args = args
        self.model = matches[0]
        self.out = args.out
        self.topology = md.load(args.top)
        self.filenames = glob.glob(os.path.join(os.path.expanduser(args.dir), '*.%s' % args.ext))
        self.featurizer = feat_load(args.featurizer)
        self.match_vars = args.match_vars
        self.stride = args.stride

        if len(self.filenames) == 0:
            self.error('No files matched.')

    def start(self):
        print('loading all data...')
        xx, ii, ff = featurize_all(
            self.filenames, self.featurizer, self.topology, self.stride)
        print('done loading')

        data = {'filename': [], 'index': [], 'state': []}
        for k in range(self.model['n_states']):
            print('computing weights for k=%d...' % k)
            try:
                weights = discrete_approx_mvn(xx, self.model['means'][k],
                                              self.model['vars'][k], self.match_vars)
            except NotSatisfiableError:
                self.error('Satisfiability failure. Could not match the means & '
                           'variances w/ discrete distribution. Try removing the '
                           'constraint on the variances with --no-match-vars?')

            cumsum = np.cumsum(weights)
            for i in range(self.args.n_per_state):
                index = int(np.sum(cumsum < np.random.rand()))
                data['filename'].append(ff[index])
                data['index'].append(ii[index])
                data['state'].append(k)

        df = pd.DataFrame(data)
        print('Saving the indices of the sampled states in CSV format to %s' % self.out)
        with open(self.out, 'w') as f:
            f.write("# command: %s\n" % ' '.join(sys.argv))
            df.to_csv(f)
