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


from __future__ import print_function, absolute_import

from ..utils import verboseload, verbosedump
from ..hiddenmarkovmodel import GaussianFusionHMM
from ..markovstatemodel import MarkovStateModel
from ..cmdline import NumpydocClassCommand, argument


class FitCommand(NumpydocClassCommand):
    inp = argument(
        '--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True)
    model = argument(
        '--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object.''', required=True)

    def start(self):
        print(self.instance)

        dataset = verboseload(self.inp)
        if not isinstance(dataset, list):
            err = '--inp must contain a list of arrays. "{}" has type {}'
            err = err.format(self.inp, type(dataset))
            self.error(err)

        print('Fitting...')
        self.instance.fit(dataset)

        verbosedump(self.instance, self.out)

        print('All done')


class GaussianFusionHMMCommand(FitCommand):
    klass = GaussianFusionHMM
    _concrete = True

class MarkovStateModelCommand(FitCommand):
    klass = MarkovStateModel
    _concrete = True