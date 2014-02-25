'''Pull structures from a CSV file
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
    filename = argument('filename', metavar='CSV_PATH',
        help='Path to csv file.')
    ext = argument('--ext', help='Output file format', default='pdb',
        choices=[e[1:] for e in md._FormatRegistry.loaders.keys()])
    top = argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)
    prefix = argument('--prefix', help='''Prefix for the output files.
        One trajectory file in the specified format will be saved with
        the name <prefix>-<state-index>.<extension>. default="state"''',
        default='state')
    

    def outfn(self, state):
        return '%s-%d.%s' % (self.prefix, state, self.ext)

    def __init__(self, args):
        self.filename = args.filename
        self.ext = args.ext
        self.prefix = args.prefix
        self.top = md.load(args.top).topology

    def start(self):
        df = pd.read_csv(self.filename, skiprows=1)
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

        for state, samples in frames.items():
            traj = samples[0].join(samples[1:])
            print('saving %s...' % self.outfn(state))
            traj.save(self.outfn(state), force_overwrite=False)
        print('done')
