'''Some reusable components for commands (mostly groups of common command
line arguments). By inheriting from one of these mixins, you get the flags
without retyping everything
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
