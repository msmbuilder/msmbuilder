'''Fit an L1-Regularized Reversible Gaussian Hidden Markov Model
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

from __future__ import print_function, division
import sys
import os
import glob
import json
import time
import numpy as np
import mdtraj as md

from sklearn.cross_validation import KFold
from mixtape.ghmm import GaussianFusionHMM
# from mixtape.lagtime import contraction
from mixtape.cmdline import Command, argument_group, MultipleIntAction
from mixtape.commands.mixins import MDTrajInputMixin, GaussianFeaturizationMixin
import mixtape.featurizer

__all__ = ['FitGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class FitGHMM(Command, MDTrajInputMixin):
    name = 'fit-ghmm'
    description = '''Fit L1-Regularized Reversible Gaussian hidden Markov models with EM.'''

    group_hmm = argument_group('HMM Options')
    group_hmm.add_argument('--featurizer', type=str, required=True,
        help='Path to saved featurizer object')
    group_hmm.add_argument('-k', '--n-states', action=MultipleIntAction, default=[2],
        help='Number of states in the models. Default = [2,]', nargs='+')
    group_hmm.add_argument('-l', '--lag-times', action=MultipleIntAction, default=[1],
        help='Lag time(s) of the model(s). Default = [1,]', nargs='+')
    group_hmm.add_argument('--platform', choices=['cuda', 'cpu', 'sklearn'],
        default='cpu', help='Implementation platform. default="cpu"')
    group_hmm.add_argument('--fusion-prior', type=float, default=1e-2,
        help='Strength of the adaptive fusion prior. default=1e-2')
    group_hmm.add_argument('--n-em-iter', type=int, default=100,
        help='Maximum number of iterations of EM. default=100')
    group_hmm.add_argument('--thresh', type=float, default=1e-2,
        help='''Convergence criterion for EM. Quit when the log likelihood
        decreases by less than this threshold. default=1e-2''')
    group_hmm.add_argument('--n-lqa-iter', type=int, default=10,
        help='''Max number of iterations for local quadradric approximation
        solving the fusion-L1. default=10''')
    group_hmm.add_argument('--reversible-type', choices=['mle', 'transpose'],
        default='mle', help='''Method by which the model is constrained to be
        reversible. default="mle"''')
    group_hmm.add_argument('-sp', '--split', type=int, help='''Split
        trajectories into smaller chunks. This looses some counts (i.e. like
        1%% of the counts are lost with --split 100), but can help with speed
        (on gpu + multicore cpu) and numerical instabilities that come when
        trajectories get extremely long.''', default=10000)

    group_cv = argument_group('Cross Validation')
    group_cv.add_argument('--n-cv', type=int, default=1,
        help='Run N-fold cross validation. default=1')
    # We're training and testing at the same lag time for the moment
    # group_cv.add_argument('--test-lag-time', type=int, default=1,
    #     help='Lag time at which to test the models. default=1')

    group_out = argument_group('Output')
    group_out.add_argument('-o', '--out', default='hmms.jsonlines',
        help='Output file. default="hmms.jsonlines"')


    def __init__(self, args):
        self.args = args
        if args.top is not None:
            self.top = md.load(os.path.expanduser(args.top))
        else:
            self.top = None

        self.featurizer = mixtape.featurizer.load(args.featurizer)
        self.filenames = glob.glob(os.path.expanduser(args.dir) + '/*.' + args.ext)
        self.n_features = self.featurizer.n_features


    def start(self):
        args = self.args
        data = self.load_data()

        with open(args.out, 'a', 0) as outfile:
            outfile.write('# %s\n' % ' '.join(sys.argv))

            for lag_time in args.lag_times:
                subsampled = [d[::lag_time] for d in data]
                for n_states in args.n_states:

                    if args.n_cv > 1:
                        for fold, (train_i, test_i) in enumerate(KFold(n=len(data), n_folds=args.n_cv)):
                            train = [subsampled[i] for i in train_i]
                            test = [subsampled[i] for i in test_i]

                            self.fit(train, test, n_states, lag_time, fold, args, outfile)
                    else:
                        self.fit(subsampled, subsampled, n_states, lag_time, 0, args, outfile)


    def fit(self, train, test, n_states, train_lag_time, fold, args, outfile):
        kwargs = dict(n_states=n_states, n_features=self.n_features, n_em_iter=args.n_em_iter,
            n_lqa_iter = args.n_lqa_iter, fusion_prior=args.fusion_prior,
            thresh=args.thresh, reversible_type=args.reversible_type,
                    platform=args.platform)
        print(kwargs)
        model = GaussianFusionHMM(**kwargs)

        start = time.time()
        model.fit(train)
        end = time.time()

        result = {
            'model': 'GaussianFusionHMM',
            'timescales': (np.real(model.timescales_()) * train_lag_time).tolist(),
            'transmat': np.real(model.transmat_).tolist(),
            'populations': np.real(model.populations_).tolist(),
            'n_states': model.n_states,
            'split': args.split,
            'fusion_prior': args.fusion_prior,
            'train_lag_time': train_lag_time,
            'train_time': end - start,
            'means': np.real(model.means_).tolist(),
            'vars': np.real(model.vars_).tolist(),
            'train_logprob': model.fit_logprob_[-1],
            'n_train_observations': sum(len(t) for t in train),
            'n_test_observations': sum(len(t) for t in test),
            'train_logprobs': model.fit_logprob_,
            #'test_lag_time': args.test_lag_time,
            'cross_validation_fold': fold,
            'cross_validation_nfolds': args.n_cv,
        }

        # model.transmat_ = contraction(model.transmat_, float(train_lag_time) / float(args.test_lag_time))
        # Don't do any contraction -- train and test at the same lagtime
        result['test_logprob'] = model.score(test)
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
                features = self.featurizer.featurize(t)
                data.append(features)

        print('Loading data into memory + vectorization: %f s' % (time.time() - load_time_start))
        print('Fitting with %s timeseries from %d trajectories with %d total observations' % (
            len(data), len(self.filenames), sum(len(e) for e in data)))

        return data
