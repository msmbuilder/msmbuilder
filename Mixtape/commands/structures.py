'''Pull structures from a CSV file
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

import os
from collections import defaultdict
import mdtraj as md
import pandas as pd
import numpy as np

from mixtape.cmdline import Command, argument

__all__ = ['Structures']


class Structures(Command):
    name = 'structures'
    description = 'Extract protein structures (.pdb/.xtc/etc) from a CSV file.'
    filename = argument('filename', metavar='CSV_PATH', help='Path to csv file.')
    ext = argument('--ext', help='Output file format', default='pdb',
        choices=[e[1:] for e in list(md._FormatRegistry.loaders.keys())])
    top = argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)
    prefix = argument('--prefix', default='state', help='''Prefix for the output
        files. One trajectory file in the specified format will be saved with
        the name <prefix>-<state-index>.<extension>. default="state"''')

    def outfn(self, state):
        return '%s-%d.%s' % (self.prefix, state, self.ext)

    def __init__(self, args):
        self.filename = args.filename
        self.ext = args.ext
        self.prefix = args.prefix
        self.top = md.load(args.top).topology

    def start(self):
        # read the csv file with an optional comment on the first line
        with open(self.filename) as f:
            line = f.readline()
            if not line.startswith('#'):
                f.seek(0, 0)
            df = pd.read_csv(f)

        if not all(e in df.columns for e in ('filename', 'index', 'state')):
            self.error('CSV file not read properly')

        for k in np.unique(df['state']):
            fn = self.outfn(k)
            if os.path.exists(fn):
                self.error('IOError: file exists: %s' % fn)

        frames = defaultdict(lambda: [])
        for fn, group in df.groupby('filename'):
            for _, row in group.sort('index').iterrows():
                frames[row['state']].append(
                    md.load_frame(fn, row['index'], top=self.top))

        for state, samples in list(frames.items()):
            traj = samples[0].join(samples[1:])
            print('saving %s...' % self.outfn(state))
            traj.save(self.outfn(state), force_overwrite=False)
        print('done')
