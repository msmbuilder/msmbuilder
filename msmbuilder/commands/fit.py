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

import os

from ..dataset import dataset
from ..utils import verbosedump
from ..hmm import GaussianFusionHMM
from ..msm import MarkovStateModel, BayesianMarkovStateModel
from ..cmdline import NumpydocClassCommand, argument, exttype


class FitCommand(NumpydocClassCommand):
    inp = argument(
        '-i', '--inp', help='''Input dataset. This should be serialized
        list of numpy arrays.''', required=True, type=os.path.expanduser)
    model = argument(
        '-o', '--out', help='''Output (fit) model. This will be a
        serialized instance of the fit model object.''', required=True,
        type=exttype('.pkl'))

    def start(self):
        if not os.path.exists(self.inp):
            self.error('File does not exist: %s' % self.inp)

        print(self.instance)
        inp_ds = dataset(self.inp, mode='r')
        self.instance.fit(inp_ds)

        print("*********\n*RESULTS*\n*********")
        print(self.instance.summarize())
        print('-' * 80)

        verbosedump(self.instance, self.out)
        print("To load this %s object interactively inside an IPython\n"
              "shell or notebook, run: \n" % self.klass.__name__)
        print("  $ ipython")
        print("  >>> from msmbuilder.utils import load")
        print("  >>> model = load('%s')\n" % self.out)

        inp_ds.close()

class GaussianFusionHMMCommand(FitCommand):
    klass = GaussianFusionHMM
    _concrete = True
    _group = 'MSM'


class MarkovStateModelCommand(FitCommand):
    klass = MarkovStateModel
    _concrete = True
    _group = 'MSM'


class BayesianMarkovStateModelCommand(FitCommand):
    klass = BayesianMarkovStateModel
    _concrete = True
    _group = 'MSM'

