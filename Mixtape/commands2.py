"""Subcommands of the `mixtape` script using the NumpydocClass command wrapper
"""
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

from __future__ import print_function
import os
import sys
import glob
import numpy as np
import mdtraj as md
from sklearn.externals.joblib import load, dump

from mixtape.tica import tICA
from mixtape.ghmm import GaussianFusionHMM
from mixtape.sparsetica import SparseTICA
from mixtape.cluster import KMeans, KCenters
from mixtape.featurizer import (ContactFeaturizer, DihedralFeaturizer,
                                AtomPairsFeaturizer, SuperposeFeaturizer,
                                DRIDFeaturizer)
from mixtape.cmdline import NumpydocClassCommand, argument

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------

def verbosedump(value, fn, compress=1):
    print('Saving "%s"... (%s)' % (fn, type(value)))
    dump(value, fn, compress=compress)

def verboseload(fn):
    print('loading "%s"...' % fn)
    return load(fn)

#-----------------------------------------------------------------------------
# Featurizer Commands
#-----------------------------------------------------------------------------

class ContactFeaturizerCommand(NumpydocClassCommand):
    klass = ContactFeaturizer
    trjs = argument('--trjs', help='Glob pattern for trajectories',
        default='', nargs='+')
    top = argument('--top', help='Path to topology file', default='')
    chunk = argument('--chunk', help='''Chunk size for loading trajectories
        using mdtraj.iterload''', default=10000, type=int)
    out = argument('--out', required=True, help='Output path')
    stride = argument('--stride', default=1, type=int, help='Load only every stride-th frame')

    def _contacts_type(self, val):
        if val is 'all':
            return val
        else:
            return np.loadtxt(fn, dtype=int, ndmin=2)

    def start(self):
        print(self.instance)
        if os.path.exists(self.top):
            top = md.load(self.top)
        else:
            top = None

        dataset = []
        for item in self.trjs:
            for trjfn in glob.glob(item):
                trajectory = []
                for i, chunk in enumerate(md.iterload(trjfn, stride=self.stride, chunk=self.chunk, top=top)):
                    print('\r{} chunk {}'.format(os.path.basename(trjfn), i), end='')
                    sys.stdout.flush()
                    trajectory.append(self.instance.featurize(chunk))
                print()
                dataset.append(np.concatenate(trajectory))

        verbosedump(dataset, self.out)
        print('All done')


class DihedralFeaturizerCommand(ContactFeaturizerCommand):
    klass = DihedralFeaturizer

class AtomPairsFeaturizerCommand(ContactFeaturizerCommand):
    klass = AtomPairsFeaturizer
    def _pair_indices_type(self, fn):
        return np.loadtxt(fn, dtype=int, ndmin=2)

class SuperposeFeaturizerCommand(ContactFeaturizerCommand):
    klass = SuperposeFeaturizer
    def _reference_traj_type(self, fn):
        return md.load(fn)
    def _atom_indices_type(self, fn):
        return np.loadtxt(fn, dtype=int, ndmin=1)

class DRIDFeaturizerCommand(ContactFeaturizerCommand):
    klass = DRIDFeaturizer
    def _atom_indices_type(self, fn):
        return np.loadtxt(fn, dtype=int, ndmin=1)



#-----------------------------------------------------------------------------
# partial_fit() on each sequence and then transform() on each sequence
#-----------------------------------------------------------------------------

class tICACommand(NumpydocClassCommand):
    klass = tICA
    inp = argument('--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    out = argument('--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object (optional).''',
        default='')
    transformed = argument('--transformed', help='''Output (transformed)
        dataset. This will be a serialized list of numpy arrays,
        corresponding to each array in the input data set after the
        applied transformation (optional).''', default='')

    def start(self):
        print(self.instance)
        if self.out is '' and self.transform is '':
            self.error('One of --out or --model should be specified')

        dataset = verboseload(self.inp)
        if not isinstance(dataset, list):
            self.error('--inp must contain a list of arrays. "%s" has type %s' % (self.inp, type(dataset)))

        for i, sequence in enumerate(dataset):
            print('partial_fit() on sequence %d shape %s...' % (i, str(sequence.shape)))
            self.instance.partial_fit(sequence)

        if self.transformed is not '':
            transformed = []
            for sequence in dataset:
                print('transform() sequence %d of shape %s...' % (i, str(sequence.shape)))
                transformed.append(self.instance.transform(sequence))
            verbosedump(transformeded, self.transformed)

        if self.out is not '':
            verbosedump(self.instance, self.out)

        print('All done')


class SparseTICACommand(tICACommand):
    klass = SparseTICA


#-----------------------------------------------------------------------------
# fit(dataset), and no transform
#-----------------------------------------------------------------------------

class GaussianFusionHMMCommand(NumpydocClassCommand):
    klass = GaussianFusionHMM
    inp = argument('--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    model = argument('--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object.''', required=True)

    def start(self):
        print(self.instance)

        dataset = verboseload(self.inp)
        if not isinstance(dataset, list):
            self.error('--inp must contain a list of arrays. "%s" has type %s' % (self.inp, type(dataset)))

        print('fitting...')
        self.instance.fit(dataset)

        verbosedump(self.instance, self.out)

        print('All done')


#-----------------------------------------------------------------------------
# fit_predict(dataset)
#-----------------------------------------------------------------------------

class KMeansCommand(NumpydocClassCommand):
    klass = KMeans
    inp = argument('--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    out = argument('--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object. (optional)''', default='')
    labels = argument('--labels', help='''Output (transformed) dataset.
        This will be a serialized list of 1D numpy arrays with the cluster
        labels of each data point in the input dataset. (optional)''', default='')

    def start(self):
        print(self.instance)
        if self.out is '' and self.labels is '':
            self.error('One of --out or --labels should be specified')

        dataset = verboseload(self.inp)
        if not isinstance(dataset, list):
            self.error('--inp must contain a list of arrays. "%s" has type %s' % (self.inp, type(dataset)))

        print('fitting...')
        labels = self.instance.fit_predict(dataset)

        if self.labels is not '':
            verbosedump(labels, self.labels)

        if self.out is not '':
            verbosedump(self.instance, self.out)

        print('All done')

class KCentersCommand(KMeansCommand):
    klass = KCenters
