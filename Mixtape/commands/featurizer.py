from __future__ import print_function, absolute_import
import os
import sys

import numpy as np

from ..cmdline import NumpydocClassCommand, argument
from ..dataset import dataset, MDTrajDataset
from ..featurizer import (AtomPairsFeaturizer, SuperposeFeaturizer,
                          DRIDFeaturizer, DihedralFeaturizer,
                          ContactFeaturizer)


class FeaturizerCommand(NumpydocClassCommand):
    trjs = argument(
        '--trjs', help='Glob pattern for trajectories',
        default='')
    top = argument(
        '--top', help='Path to topology file', default='')
    chunk = argument(
        '--chunk',
        help='''Chunk size for loading trajectories using mdtraj.iterload''',
        default=10000, type=int)
    out = argument(
        '--out', required=True, help='Output path')
    stride = argument(
        '--stride', default=1, type=int,
        help='Load only every stride-th frame')


    def start(self):
        print(self.instance)
        if os.path.exists(self.top):
            top = self.top
        else:
            top = None

        input_dataset = MDTrajDataset(self.trjs, topology=top, stride=self.stride, verbose=True)
        out_dataset = input_dataset.write_derived(self.out, [], fmt='dir-npy')

        for key in input_dataset.keys():
            trajectory = []
            for i, chunk in enumerate(input_dataset.iterload(key, chunk=self.chunk)):
                print('\r{chunk %d}' % i, end='')
                sys.stdout.flush()
                trajectory.append(self.instance.partial_transform(chunk))
            print()
            out_dataset[key] = np.concatenate(trajectory)

        print('All done')


class DihedralFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = DihedralFeaturizer


class AtomPairsFeaturizerCommand(FeaturizerCommand):
    klass = AtomPairsFeaturizer
    _concrete = True

    def _pair_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=2)


class SuperposeFeaturizerCommand(FeaturizerCommand):
    klass = SuperposeFeaturizer
    _concrete = True

    def _reference_traj_type(self, fn):
        return md.load(fn)

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class DRIDFeaturizerCommand(FeaturizerCommand):
    klass = DRIDFeaturizer
    _concrete = True

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class ContactFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = ContactFeaturizer

    def _contacts_type(self, val):
        if val is 'all':
            return val
        else:
            return np.loadtxt(val, dtype=int, ndmin=2)
