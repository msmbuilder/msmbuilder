from __future__ import print_function, division
import pandas as pd
from textwrap import wrap

from mixtape.utils import iterobjects
from mixtape.cmdline import Command, argument, argument_group

__all__ = ['Jsonlines2CSV']


class Jsonlines2CSV(Command):
    name = 'jsonlines2csv'
    description = '''Save a small CSV summary of one or more jsonlines files.

    Optionally, we can also make some small plots'''
    filename = argument('filename', nargs="+", metavar='JSONLINES_FILE', help='''
        Path to .jsonlines file''')
    out = argument('out_csv', metavar='OUT_CSV_FILE', help='')

    g = argument_group("optional plotting options (if supplied, a plot is displayed too)")
    x = g.add_argument('--x')
    y = g.add_argument('--y')
    plot_color = g.add_argument('--color')

    def __init__(self, args):
        self.args = args

    def start(self):
        exclude = set([
            'means', 'vars', 'kappas', 'train_logprobs', 'transmat', 'split',
            'test_logprob', 'n_test_observations', 'fusion_prior',
            'test_lag_time', 'cross_validation_fold', 'train_time',
            'cross_validation_nfolds'])
        models = [{k: v for k, v in m.items() if k not in exclude}
                  for f in self.args.filename for m in iterobjects(f)]

        explode_key_in_listofdicts('timescales', models)
        explode_key_in_listofdicts('populations', models)
        self.models = pd.DataFrame(models)

        print('Columns:\n    %s' % '\n    '.join(wrap(', '.join(self.models.columns))))
        print('Saving models as CSV file to "%s"' % self.args.out_csv)
        self.models.to_csv(self.args.out_csv, index=False)

        if self.args.x is not None and self.args.y is not None:
            self.plot()

    def plot(self):
        if not self.args.x in self.models.columns:
            self.error('--x must be one of %s' % ', '.join(self.models.columns))
        if not self.args.y in self.models.columns:
            self.error('--y must be one of %s' % ', '.join(self.models.columns))
        if not self.args.color in self.models.columns:
            self.error('--color must be one of %s' % ', '.join(self.models.columns))

        import matplotlib.pyplot as pp
        if self.args.color is not None:
            for key, group in self.models.groupby(self.args.color):
                pp.plot(group[self.args.x], group[self.args.y], 'x', label='%s=%s' %
                        (self.args.color, key))
            pp.legend(loc=2)
        else:
            pp.plot(self.models[self.args.x], self.models[self.args.y], marker='x')

        pp.xlabel(self.args.x)
        pp.ylabel(self.args.y)
        pp.title('%s vs %s' % (self.args.y, self.args.x))
        pp.show()


def explode_key_in_listofdicts(key, listofdicts, default=float('NaN')):
    length = max(len(d[key]) for d in listofdicts)
    for d in listofdicts:
        for i in range(length):
            d['%s-%d' % (key, i)] = d[key][i] if i < len(d[key]) else default
        del d[key]
