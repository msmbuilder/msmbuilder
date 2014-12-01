from __future__ import print_function, division, absolute_import
import os
import sys
import json

import pandas as pd
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from ..dataset import dataset
from ..cmdline import Command, argument, argument_group, slicetype, FlagAction
from ..msm import MarkovStateModel


class ImpliedTimescales(Command):
    _group = 'MSM'
    _concrete = True
    description = "Scan the implied timescales of `MarkovStateModel`s with respect to lag time"
    lag_times = argument('-l', '--lag_times', default='1:10', type=slicetype, help='''
        Range of lag times. Specify as 'start:stop' or 'start:stop:step''')
    inp = argument(
        '--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    out = argument('--out', help='''Output file''',
        default='timescales.csv')
    fmt = argument('--fmt', help='Output file format', default='csv',
        choices=('csv', 'json', 'excel'))
    _extensions = {'csv': '.csv', 'json': '.json', 'excel': '.xlsx'}

    n_jobs = argument('--n_jobs', help='Number of parallel processes', default=1)

    p = argument_group('Parameters')
    n_timescales = p.add_argument('--n_timescales', default=10, help='''
        The number of dynamical timescales to calculate when diagonalizing
        the transition matrix.''',  type=int)
    reversible_type = p.add_argument('--reversible_type', default='mle', help='''
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization, and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.''',
        choices=('mle', 'transpose'))
    ergodic_cutoff = p.add_argument('--ergodic_cutoff', default=1, type=int, help='''
        Only the maximal strongly ergodic subgraph of the data is used to build
        an MSM. Ergodicity is determined by ensuring that each state is
        accessible from each other state via one or more paths involving edges
        with a number of observed directed counts greater than or equal to
        ``ergodic_cutoff``. Not that by setting ``ergodic_cutoff`` to 0, this
        trimming is effectively turned off.''')
    prior_counts = p.add_argument('--prior_counts', default=0, help='''Add a number
        of "pseudo counts" to each entry in the counts matrix. When
        prior_counts == 0 (default), the assigned transition probability
        between two states with no observed transitions will be zero, whereas
        when prior_counts > 0, even this unobserved transitions will be
        given nonzero probability.''', type=float)
    verbose = p.add_argument('--verbose', default=True,
        help='Enable verbose printout', action=FlagAction)

    def __init__(self, args):
        self.args = args

    def start(self):
        parallel = Parallel(n_jobs=self.args.n_jobs, verbose=self.args.verbose)
        ds = dataset(self.args.inp, mode='r', fmt='dir-npy')
        kwargs = {
            'n_timescales': self.args.n_timescales,
            'reversible_type': self.args.reversible_type,
            'ergodic_cutoff': self.args.ergodic_cutoff,
            'prior_counts': self.args.prior_counts,
            'verbose': self.args.verbose,
        }

        lines = parallel(delayed(_fit_and_timescales)(
                lag_time, kwargs, ds)
            for lag_time in range(*self.args.lag_times.indices(sys.maxsize)))

        cols = ['Lag time',] + ['Timescale %d' % (d+1) for d in range(len(lines[0])-1)]

        df = pd.DataFrame(data=lines, columns=cols)
        self.write_output(df)

    def write_output(self, df):
        outfile = os.path.splitext(self.args.out)[0] + self._extensions[self.args.fmt]

        print('Writing %s' % outfile)
        if self.args.fmt == 'csv':
            df.to_csv(outfile)
        elif self.args.fmt == 'json':
            with open(outfile, 'w') as f:
                json.dump(df.to_dict(orient='records'), f)
        elif self.args.fmt == 'excel':
            df.to_excel(outfile)
        print('All done!')


def _fit_and_timescales(lag_time, kwargs, ds):
    model = MarkovStateModel(lag_time=lag_time, **kwargs)
    model.fit(ds)
    return ((lag_time, ) + tuple(model.timescales_))
