'''Some reusable components for commands (mostly groups of common command
line arguments). By inheriting from one of these mixins, you get the flags
without retyping everything
'''
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

from __future__ import print_function, division, absolute_import

import mdtraj as md
from mixtape.cmdline import argument_group

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class MDTrajInputMixin(object):

    """Mixin for a command to accept trajectory input files"""
    group_mdtraj = argument_group('MDTraj Options')
    group_mdtraj.add_argument('--dir', type=str, help='''Directory containing
        the trajectories to load''', required=True)
    group_mdtraj.add_argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)
    group_mdtraj.add_argument('--ext', help='File extension of the trajectories',
        required=True, choices=[e[1:] for e in list(md._FormatRegistry.loaders.keys())])


class GaussianFeaturizationMixin(object):
    group_munge = argument_group('Featurizer Options')
    group_munge.add_argument('-d', '--distance-pairs', type=str, help='''Vectorize
        the MD trajectories by extracting timeseries of the distance
        between pairs of atoms in each frame. Supply a text file where
        each row contains the space-separate indices of two atoms which
        form a pair to monitor''')
    group_munge.add_argument('-a', '--atom-indices', type=str, help='''Superpose
        each MD conformation on the coordinates in the topology file, and then use
        the distance from each atom in the reference conformation to the
        corresponding atom in each MD conformation.''')
    group_munge.add_argument('-s', '--solvent-indices', type=str,
        help='''Calculate 'solvent fingerprint' by summing weighted distances
        between each solute atom and all solvent atoms. Supply a text file
        of solvent indices. You must also specify --atom-indices for
        the solute atoms.''')
    group_munge.add_argument('--sigma', type=float,
        help='''If --solvent-indices is specified, this sets the length scale
        for the Gaussian kernel in the solvent fingerprint. Otherwise ignored.''')
