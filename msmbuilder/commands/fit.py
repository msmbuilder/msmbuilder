# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2014, Stanford University
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import print_function, absolute_import

import os

from ..dataset import dataset
from ..utils import verbosedump
from ..hmm import GaussianHMM
from ..msm import (MarkovStateModel, BayesianMarkovStateModel, ContinuousTimeMSM,
                   BayesianContinuousTimeMSM)
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

class GaussianHMMCommand(FitCommand):
    klass = GaussianHMM
    _concrete = True
    _group = 'MSM'


class MarkovStateModelCommand(FitCommand):
    klass = MarkovStateModel
    _concrete = True
    _group = 'MSM'

    def _ergodic_cutoff_type(self, erg):
        if erg.lower() in ['on', 'off']:
            return erg
        else:
            return float(erg)


class BayesianMarkovStateModelCommand(FitCommand):
    klass = BayesianMarkovStateModel
    _concrete = True
    _group = 'MSM'


class ContinuousTimeMSMCommand(FitCommand):
    klass = ContinuousTimeMSM
    _concrete = True
    _group = 'MSM'


class BayesianContinuousTimeMSMCommand(FitCommand):
    klass = BayesianContinuousTimeMSM
    _concrete = True
    _group = 'MSM'
