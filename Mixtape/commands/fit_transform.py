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

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import print_function, absolute_import

import os

from ..utils.progressbar import ProgressBar, Percentage, Bar, ETA
from ..dataset import dataset
from ..utils import verbosedump
from ..decomposition import tICA, PCA
from ..cluster import (KMeans, KCenters, KMedoids, MiniBatchKMedoids,
                       MiniBatchKMeans)
from ..cmdline import NumpydocClassCommand, argument, argument_group, exttype


class FitTransformCommand(NumpydocClassCommand):
        # What format to use for saving transformed dataset
    _transformed_fmt = 'hdf5'

    g1 = argument_group('input')
    inp = g1.add_argument(
        '-i', '--inp', help='''Path to input dataset. Each dataset is a
        collection of sequences, which may either be an array or Trajectory,
        depending on the model.''', required=True)
    g2 = argument_group('output', description='use one or both of the following')
    out = g2.add_argument(
        '-o', '--out', help='''Path to save fit model instance. This is the
        model, after parameterization with fit(), saved using the pickle
        protocol''',
        default='', type=exttype('.pkl'))
    transformed = g2.add_argument(
        '-t', '--transformed', help='''Path to output transformed dataset. This
        will be a collection of arrays, as transfomed by the model''',
        default='', type=exttype('.h5'))

    def start(self):
        if self.out is '' and self.transformed is '':
            self.error('One of --out or --model should be specified')
        if self.transformed is not '' and os.path.exists(self.transformed):
            self.error('File exists: %s' % self.transformed)

        print(self.instance)

        inp_ds = dataset(self.inp, mode='r', verbose=False)
        print("Fitting model...")
        self.instance.fit(inp_ds)

        print("*********\n*RESULTS*\n*********")
        print(self.instance.summarize())
        print('-' * 80)

        if self.transformed is not '':
            out_ds = inp_ds.create_derived(self.transformed, fmt=self._transformed_fmt)
            pbar = ProgressBar(
                widgets=['Transforming ', Percentage(), Bar(), ETA()],
                maxval=len(inp_ds)).start()

            for key in pbar(inp_ds.keys()):
                in_seq = inp_ds.get(key)
                out_ds[key] = self.instance.partial_transform(in_seq)

            print("\nSaving transformed dataset to '%s'" % self.transformed)
            print("To load this dataset interactive inside an IPython")
            print("shell or notebook, run\n")
            print("  $ ipython")
            print("  >>> from mixtape.dataset import dataset")
            print("  >>> ds = dataset('%s')\n" % self.transformed)

        if self.out is not '':
            verbosedump(self.instance, self.out)
            print("To load this %s object interactively inside an IPython\n"
                  "shell or notebook, run: \n" % self.klass.__name__)
            print("  $ ipython")
            print("  >>> from mixtape.utils import load")
            print("  >>> model = load('%s')\n" % self.out)


class tICACommand(FitTransformCommand):
    klass = tICA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class PCACommand(FitTransformCommand):
    klass = PCA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class KMeansCommand(FitTransformCommand):
    klass = KMeans
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'

    def _random_state_type(self, state):
        if state is None:
            return None
        return int(state)


class MiniBatchKMeansCommand(KMeansCommand):
    klass = MiniBatchKMeans
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class KCentersCommand(KMeansCommand):
    klass = KCenters
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class KMedoidsCommand(KMeansCommand):
    klass = KMedoids
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class MiniBatchKMedoidsCommand(KMeansCommand):
    klass = MiniBatchKMedoids
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'

