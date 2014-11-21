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

from ..utils import verboseload, verbosedump
from ..decomposition import tICA, PCA
from ..cluster import KMeans, KCenters, KMedoids, MiniBatchKMedoids, MiniBatchKMeans
from ..cmdline import NumpydocClassCommand, argument


class FitTransformCommand(NumpydocClassCommand):
    inp = argument(
        '--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    out = argument(
        '--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object (optional).''',
        default='')
    transformed = argument(
        '--transformed', help='''Output (transformed)
        dataset. This will be a serialized list of numpy arrays,
        corresponding to each array in the input data set after the
        applied transformation (optional).''', default='')

    def start(self):
        print(self.instance)
        if self.out is '' and self.transform is '':
            self.error('One of --out or --model should be specified')

        dataset = verboseload(self.inp)
        if not isinstance(dataset, list):
            err = '--inp must contain a list of arrays. "{}" has type {}'
            err = err.format(self.inp, type(dataset))
            self.error(err)

        fstr = 'Performing fit() on {} sequences of shape {}...'
        shapes = [str(dataset[i].shape) for i in range(min(3, len(dataset)))]
        fval = (len(dataset), ', '.join(shapes))
        print(fstr.format(*fval))
        self.instance.fit(dataset)

        if self.transformed is not '':
            transformed = self.instance.transform(dataset)
            verbosedump(transformed, self.transformed)

        if self.out is not '':
            verbosedump(self.instance, self.out)

        print('All done')


class tICACommand(FitTransformCommand):
    klass = tICA
    _concrete = True
    _group = '3-Decomposition'

class PCACommand(FitTransformCommand):
    klass = PCA
    _concrete = True
    _group = '3-Decomposition'


class KMeansCommand(FitTransformCommand):
    klass = KMeans
    _concrete = True
    _group = '2-Clustering'

    def _random_state_type(self, state):
        if state is None:
            return None
        return int(state)


class MiniBatchKMeansCommand(KMeansCommand):
    klass = MiniBatchKMeans
    _concrete = True
    _group = '2-Clustering'


class KCentersCommand(KMeansCommand):
    klass = KCenters
    _concrete = True
    _group = '2-Clustering'


class KMedoidsCommand(KMeansCommand):
    klass = KMedoids
    _concrete = True
    _group = '2-Clustering'


class MiniBatchKMedoidsCommand(KMeansCommand):
    klass = MiniBatchKMedoids
    _concrete = True
    _group = '2-Clustering'
