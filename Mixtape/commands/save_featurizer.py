'''Fit an L1-Regularized Reversible Gaussian Hidden Markov Model
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division
import sys
import os
import glob
import json
import time
import numpy as np
import mdtraj as md

from sklearn.cross_validation import KFold
from mixtape.ghmm import GaussianFusionHMM
# from mixtape.lagtime import contraction
from mixtape.cmdline import Command, argument_group, MultipleIntAction
from mixtape.commands.mixins import MDTrajInputMixin, GaussianFeaturizationMixin
from mixtape.featurizer import SuperposeFeaturizer, AtomPairsFeaturizer

__all__ = ['SaveFeaturizer']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class SaveFeaturizer(Command, GaussianFeaturizationMixin):
    name = 'save-featurizer'
    description = '''Save a featurizer for later use'''

    group_feature = argument_group('Featurizer Loading')
    group_feature.add_argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)    
    group_feature.add_argument('-f', '--filename', type=str, help='''Output
    featurizer to this filename.''')


    def __init__(self, args):
        self.args = args
        if args.top is not None:
            self.top = md.load(os.path.expanduser(args.top))
        else:
            self.top = None

        if args.distance_pairs is not None:
            self.indices = np.loadtxt(args.distance_pairs, dtype=int, ndmin=2)
            if self.indices.shape[1] != 2:
                self.error('distance-pairs must have shape (N, 2). %s had shape %s' % (args.distance_pairs, self.indices.shape))
            featurizer = AtomPairsFeaturizer(self.indices, self.top)                
        else:
            self.indices = np.loadtxt(args.atom_indices, dtype=int, ndmin=2)
            if self.indices.shape[1] != 1:
                self.error('atom-indices must have shape (N, 1). %s had shape %s' % (args.atom_indices, self.indices.shape))
            self.indices = self.indices.reshape(-1)
            
            featurizer = SuperposeFeaturizer(self.indices, self.top)
            
        featurizer.save(args.filename)
        

    def start(self):
        args = self.args

