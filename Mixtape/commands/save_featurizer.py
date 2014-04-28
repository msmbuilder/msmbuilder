'''Save a trajectory featurizer to disk
'''
# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
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

from __future__ import print_function, division, absolute_import

import os
import numpy as np
import mdtraj as md

from mixtape.cmdline import Command, argument_group
from mixtape.commands.mixins import GaussianFeaturizationMixin
from mixtape.featurizer import SuperposeFeaturizer, AtomPairsFeaturizer

__all__ = ['SaveFeaturizer']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SaveFeaturizer(Command, GaussianFeaturizationMixin):
    name = 'featurizer'
    description = '''Create and save a featurizer for later use.'''

    group_feature = argument_group('Featurizer Loading')
    group_feature.add_argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)
    group_feature.add_argument('-o', '--filename', default='featurizer.pickl', help='''
        Output featurizer to this filename. default="featurizer.pickl"''')

    def __init__(self, args):
        self.args = args

    def start(self):
        args = self.args
        if args.top is not None:
            self.top = md.load(os.path.expanduser(args.top))
        else:
            self.top = None

        if args.distance_pairs is not None:
            self.indices = np.loadtxt(args.distance_pairs, dtype=int, ndmin=2)
            if self.indices.shape[1] != 2:
                self.error('distance-pairs must have shape (N, 2). %s had shape %s' %
                           (args.distance_pairs, self.indices.shape))
            featurizer = AtomPairsFeaturizer(self.indices, self.top)
        else:
            self.indices = np.loadtxt(args.atom_indices, dtype=int, ndmin=2)
            if self.indices.shape[1] != 1:
                self.error('atom-indices must have shape (N, 1). %s had shape %s' %
                           (args.atom_indices, self.indices.shape))
            self.indices = self.indices.reshape(-1)

            featurizer = SuperposeFeaturizer(self.indices, self.top)

        featurizer.save(args.filename)
