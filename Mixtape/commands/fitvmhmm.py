'''Fit an L1-Regularized Reversible Von-Mises Hidden Markov Model
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function

import sys
import glob
import json
import time
import numpy as np
import mdtraj as md

from mixtape.vmhmm import VonMisesHMM
from mixtape.cmdline import Command, argument_group, MultipleIntAction
from mixtape.commands.mixins import MDTrajInputMixin

__all__ = ['FitVMHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class FitVMHMM(Command, MDTrajInputMixin):
    name = 'fit-vmhmm'
    description = '''Fit von-Mises hidden Markov models with EM.

    The von Mises distribution, (also known as the circular normal
    distribution or Tikhonov distribution) is a continuous probability
    distribution on the circle. For multivariate signals, the emissions
    distribution implemented by this model is a product of univariate
    von Mises distributions -- analogous to the multivariate Gaussian
    distribution with a diagonal covariance matrix.
    
    Because the support of the base 1D distribution is on [-pi, pi), this
    model makes a suitable emission distribution for timeseries of angles
    (e.g. protein dihedral angles).
    '''

    group_munge = argument_group('Munging Options')
    group_munge.add_argument('-d', '--dihedral-indices', required=True, type=str,
        help='''Vectorize the MD trajectories by extracting timeseries of the
        dihedral (torsion) angles between sets of 4 atoms. Supply a text file
        where each row contains the space-separate indices of four atoms which
        form a dihedral angle to monitor. These indices are 0-based.''')

    group_hmm = argument_group('HMM Options')
    group_hmm.add_argument('-k', '--n-states', action=MultipleIntAction, default=[2],
        help='Number of states in the models. Default = [2,]', nargs='+')
    group_hmm.add_argument('-l', '--lag-times', action=MultipleIntAction, default=[1],
        help='Lag time(s) of the model(s). Default = [1,]', nargs='+')
    # group_hmm.add_argument('--platform', choices=['cuda', 'cpu', 'sklearn'],
    #     default='cpu', help='Implementation platform. default="cpu"')
    # group_hmm.add_argument('--fusion-prior', type=float, default=1e-2,
    #    help='Strength of the adaptive fusion prior. default=1e-2')
    group_hmm.add_argument('--n-em-iter', type=int, default=100,
        help='Maximum number of iterations of EM. default=100')
    group_hmm.add_argument('--thresh', type=float, default=1e-2,
        help='''Convergence criterion for EM. Quit when the log likelihood
        decreases by less than this threshold. default=1e-2''')
    # group_hmm.add_argument('--n-lqa-iter', type=int, default=10,
    #     help='''Max number of iterations for local quadradric approximation
    #    solving the fusion-L1. default=10''')
    group_hmm.add_argument('--reversible-type', choices=['mle', 'transpose'],
        default='mle', help='''Method by which the model is constrained to be
        reversible. default="mle"''')
    group_hmm.add_argument('-sp', '--split', type=int, help='''Split
            trajectories into smaller chunks. This looses some counts (i.e. like
            1%% of the counts are lost with --split 100), but can help with speed
            (on gpu + multicore cpu) and numerical instabilities that come when
            trajectories get extremely long.''', default=10000)

    group_out = argument_group('Output')
    group_out.add_argument('-o', '--out', default='hmms.jsonlines',
        help='Output file. default="hmms.jsonlines"')
        
    def __init__(self, args):
        self.args = args
        self.top = md.load(args.top) if args.top is not None else None

        self.indices = np.loadtxt(args.dihedral_indices, dtype=int, ndmin=2)
        if self.indices.shape[1] != 4:
            self.error('dihedral-indices must have shape (N, 4). %s had shape %s' % (args.dihedral_indices, self.indices.shape))
        self.filenames = glob.glob(args.dir + '/*.' + args.ext)
        self.n_features = self.indices.shape[0]
        
    def start(self):
        args = self.args
        data = self.load_data()

        with open(args.out, 'a', 0) as outfile:
            outfile.write('# %s\n' % ' '.join(sys.argv))

            for lag_time in args.lag_times:
                subsampled = [d[::lag_time] for d in data]
                for n_states in args.n_states:
                    self.fit(subsampled, n_states, lag_time, outfile)
    
    def fit(self, data, n_states, lag_time, outfile):
        model = VonMisesHMM(n_states=n_states,
            reversible_type=self.args.reversible_type,
            n_iter=self.args.n_em_iter, thresh=self.args.thresh)
        start = time.time()
        model.fit(data)
        end = time.time()

        result = {
            'model': 'VonMisesHMM',
            'timescales': (model.timescales_() * lag_time).tolist(),
            'transmat': model.transmat_.tolist(),
            'populations': model.populations_.tolist(),
            'n_states': model.n_states,
            'split': self.args.split,
            'train_lag_time': lag_time,
            'train_time': end - start,
            'means': model.means_.tolist(),
            'kappas': model.kappas_.tolist(),
            'train_logprobs': model.fit_logprob_,
            'n_train_observations': sum(len(t) for t in data),
        }
        if not np.all(np.isfinite(model.transmat_)):
            print('Nonfinite numbers in transmat !!')
        json.dump(result, outfile)
        outfile.write('\n')

    def load_data(self):
        load_time_start = time.time()
        data = []
        for tfn in self.filenames:
            kwargs = {} if tfn.endswith('h5') else {'top': self.top}
            for t in md.iterload(tfn, chunk=self.args.split, **kwargs):
                item = np.asarray(md.compute_dihedrals(t, self.indices), np.double)
                data.append(item)
        return data


                    
