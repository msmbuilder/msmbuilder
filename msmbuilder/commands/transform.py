# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
'''
Use a pre-fit model to transform a dataset.

'''

from __future__ import print_function, absolute_import

import os

from ..base import BaseEstimator
from ..utils import load
from ..dataset import dataset
from ..cmdline import Command, argument, argument_group, exttype, stripquotestype


class TransformCommand(Command):
    _concrete = True
    _group = 'X'
    name = 'TransformDataset'
    description = __doc__

    g = argument_group('required arguments')
    inp = g.add_argument(
        '-i', '--inp', help='''Path to input dataset. Each dataset is a
        collection of sequences, which may either be an array or Trajectory,
        depending on the model.''', required=True, type=stripquotestype)
    mdl = g.add_argument('-m', '--model', help='''Path pre-fit model instance,
        saved using the pickle protocol. (suffix .pkl)''', required=True)
    transformed = g.add_argument(
        '-t', '--transformed', help='''Path to output transformed dataset. This
        will be a collection of arrays, as transformed by the model''',
        default='', type=exttype('.h5'), required=True)

    md = argument_group('mdtraj input', description='additional options '
        'required for loading trajectory datasets')
    md.add_argument('--stride', default=1, help='Load only every stride-th frame')
    md.add_argument('--top', default='', help='Path to topology file matching the trajectories')
    md.add_argument('--atom_indices', help='''Path to an index file
        containing the zero-based indices of the atoms to use for RMSD''')

    def __init__(self, args):
        self.args = args

    def start(self):

        # load the model
        model = load(self.args.model)
        if not isinstance(model, BaseEstimator):
            self.error('%r is not an MSMBuilder model' % model)
        print(model.summarize())
        print('-' * 25, '\n')

        if os.path.exists(self.args.transformed):
            self.error('File exists:' % self.args.transformed)

        # load the input dataset
        print('Opening dataset %s...' % self.args.inp)
        with dataset(self.args.inp, mode='r', topology=self.args.top,
                     stride=self.args.stride,
                     atom_indices=self.args.atom_indices) as inp_ds:
            # create the output dataset
            print('Writing to %s...' % self.args.transformed)
            with inp_ds.create_derived(self.args.transformed, fmt='hdf5') as out_ds:

                if hasattr(model, 'partial_transform'):
                    print('Calling %s.partial_transform()...' %
                          model.__class__.__name__)
                    for key in inp_ds.keys():
                        out_ds[key] = model.partial_transform(inp_ds[key])
                else:
                    print('Calling %s.transform()...' %
                          model.__class__.__name__)
                    for key, seq in zip(inp_ds.keys(), model.transform(inp_ds)):
                        out_ds[key] = seq

        print('\nAll done!')
