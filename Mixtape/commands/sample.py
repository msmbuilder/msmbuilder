'''Sample from each state in an HMM
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function

import os
import glob
import numpy as np
import mdtraj as md
import pandas as pd

from mixtape.utils import iterobjects, load_superpose_timeseries
from mixtape.discrete_approx import discrete_approx_mvn
from mixtape.cmdline import Command, argument_group
from mixtape.commands.mixins import MDTrajInputMixin, GaussianFeaturizationMixin

__all__ = ['SampleGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class SampleGHMM(Command, MDTrajInputMixin, GaussianFeaturizationMixin):
    name = 'sample-ghmm'
    description = '''Draw iid samples from each state in a Gaussian HMM.

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

    group = argument_group('I/O Arguments')
    group.add_argument('--filename', required=True, help='''Path to the
        jsonlines output file containg the HMMs''')
    group.add_argument('--n-states', type=int, required=True, help='''Number of
        states in the model to select from''')
    group.add_argument('--n-per-state', type=int, default=3, help='''Number of
        structures to pull from each state''')
    group.add_argument('--lag-time', type=int, required=True, help='''Training lag
        time of the model to select from''')
    group.add_argument('-o', '--out', metavar='OUTPUT_CSV_FILE', required=True,
        help='File to which to save the output, in CSV format')

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
            for i in range(self.args.n_per_state):
                index = np.sum(cumsum < np.random.rand())
                data['filename'].append(ff[index])
                data['index'].append(ii[index])
                data['state'].append(k)

        df = pd.DataFrame(data)
        print('Saving the indices of the sampled states in CSV format to %s' % self.out)
        df.to_csv(self.out)
