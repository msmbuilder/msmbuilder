'''Fit a Metastable Switching Linear Dynamical System.
'''
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
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
import sys
import os
import glob
import json
import time
import numpy as np
import mdtraj as md
import traceback, sys, code, pdb

from mixtape.mslds import MetastableSwitchingLDS
from mixtape.cmdline import Command, argument_group, MultipleIntAction
from mixtape.commands.mixins import MDTrajInputMixin
import mixtape.featurizer

__all__ = ['FitMSLDS']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class FitMSLDS(Command, MDTrajInputMixin):
    name = 'fit-mslds'
    description = '''Fit Metastable Switching Linear Dynamical System with
    Generalized-EM.'''

    group_mslds = argument_group('MSLDS Options')
    group_mslds.add_argument('--featurizer', type=str, required=True, help='''
        Path to saved featurizer object''')
    group_mslds.add_argument('-k', '--n-states', action=MultipleIntAction, default=[2], help='''
        Number of states in the models. Default = [2,]''', nargs='+')
    group_mslds.add_argument('-l', '--lag-times', action=MultipleIntAction, default=[1], help='''
        Lag time(s) of the model(s). Default = [1,]''', nargs='+')
    group_mslds.add_argument('--platform', choices=['cpu'], default='cpu', help='''
        Implementation platform. default="cpu"''')
    group_mslds.add_argument('--n-init', type=int, default=5, help=
    '''Number of initialization for each model fit. Each of these
        "outer iterations" corresponds to a new random initialization of
        the states from kmeans and then `--n-hotstart` iterations of HMM
        hotstart and then `--n-em-iter` minus `--n-hotstart` iterations of
        expectation-maximization. The best of these
        models (selected by likelihood) is retained. default=5''')
    group_mslds.add_argument('--n-em-iter', type=int, default=10, help='''
        Maximum number of iterations of EM. default=10''')
    group_mslds.add_argument('--max-iters', type=int, default=10, help='''
        Maximum number of SDP solver iterations per EM step. default=10''')
    group_mslds.add_argument('--n-hotstart', type=int, default=5, help='''
        Maximum number of HMM hot-starting iterations of EM. default=5''')
    group_mslds.add_argument('--reversible-type', choices=['mle'], default='mle', help='''
        Method by which the model is constrained to be reversible. default="mle"''')
    group_mslds.add_argument('-sp', '--split', type=int, help='''Split
        trajectories into smaller chunks. This loses some counts (i.e. like
        1%% of the counts are lost with --split 100), but can help with
        speed (on gpu + multicore cpu) and numerical instabilities that
        come when trajectories get extremely long.''', default=10000)
    group_mslds.add_argument('--use-pdb', action='store_true',
		help= '''Launch python debugger PDB on exception. Useful for debugging.''')
    group_mslds.add_argument('--display-solver-output', action='store_true',
        help='''Display the output of the SDP solvers.''')

    group_out = argument_group('Output')
    group_out.add_argument('-o', '--out', default='mslds.jsonlines',
                           help='Output file. default="mslds.jsonlines"')

    def __init__(self, args):
        self.args = args
        if args.top is not None:
            self.top = md.load(os.path.expanduser(args.top))
        else:
            self.top = None

        self.featurizer = mixtape.featurizer.load(args.featurizer)
        self.filenames = glob.glob(os.path.expanduser(args.dir) + '/*.' + args.ext)
        self.n_features = self.featurizer.n_features
        print("n_features = %d" % self.n_features)

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
        args = self.args
        data = self.load_data()

        with open(args.out, 'a', 0) as outfile:
            outfile.write('# %s\n' % ' '.join(sys.argv))

            for lag_time in args.lag_times:
                subsampled = [d[::lag_time] for d in data]
                for n_states in args.n_states:
                    self.fit(subsampled, n_states,
                             lag_time, 0, args, outfile)

    def fit(self, train, n_states, train_lag_time, fold, args, outfile):
        kwargs = dict(n_states=n_states, n_features=self.n_features,
                      n_init=args.n_init, max_iters=args.max_iters,
                      n_em_iter=args.n_em_iter,
                      n_hotstart=args.n_hotstart,
                      reversible_type=args.reversible_type,
                      platform=args.platform,
                      display_solver_output=args.display_solver_output)
        print(kwargs)
        model = MetastableSwitchingLDS(**kwargs)

        start = time.time()
        model.fit(train)
        end = time.time()

        result = {
            'model': 'MetastableSwitchingLinearDynamicalSystem',
            'transmat': model.transmat_.tolist(),
            'populations': model.populations_.tolist(),
            'n_states': model.n_states,
            'split': args.split,
            'train_lag_time': train_lag_time,
            'train_time': end - start,
            'n_features': model.n_features,
            'means': model.means_.tolist(),
            'covars': model.covars_.tolist(),
            'As': model.As_.tolist(),
            'bs': model.bs_.tolist(),
            'Qs': model.Qs_.tolist(),
            'train_logprob': model.fit_logprob_[-1],
            'n_train_observations': sum(len(t) for t in train),
            'train_logprobs': model.fit_logprob_,
        }

        #result['test_logprob'] = model.score(test)
        result['test_lag_time'] = train_lag_time

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
                features = self.featurizer.partial_transform(t)
                data.append(features)

        print('Loading data into memory + vectorization: %f s' %
              (time.time() - load_time_start))
        print('''Fitting with %s timeseries from %d trajectories with %d
                total observations''' % (len(data), len(self.filenames),
                                         sum(len(e) for e in data)))
        return data
