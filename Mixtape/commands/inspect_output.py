'''Inspect the contents of a .jsonlines file
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division

import sys
import numpy as np
import pandas as pd

from mixtape.utils import iterobjects
from mixtape.cmdline import Command, argument

__all__ = ['Inspect']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class Inspect(Command):
    description = "Inspect the content of a '.jsonlines' output file"
    input = argument('-i', '--filename', metavar='JSONLINES_FILE',
        required=True, help='Path to .jsonlines file')
    details = argument('--details', action='store_true', help='Print all of the details of each model to stdout')
    n_states = argument('--n-states', type=int, help='Filter the output, only showing models with the specified n_states')
    lag_time = argument('--lag-time', type=int, help='Filter the output, only showing models with the specified lag_time')

    def __init__(self, args):
        self.args = args
        self.models = list(iterobjects(self.args.filename))
        self.df = pd.DataFrame(self.models)
        self.details = args.details

    def start(self):
        df = self.df
        if self.args.n_states:
            df = df[df['n_states'] == self.args.n_states]
        if self.args.lag_time:
            df = df[df['train_lag_time'] == self.args.lag_time]
            
        # print csv to stdout
        print('-'*80)
        print('Overview')
        print('-'*80)

        df[['train_lag_time', 'n_states', 'timescales']].to_csv(sys.stdout, sep='\t', index=False)

        if self.details:
            print('-'*80)
            print('Details')
            print('-'*80)
            for i in range(len(df)):
                print('N training observations:', df['n_train_observations'][i])
                print('\nTrain Lag Time:', df['train_lag_time'][i])
                print('\nN States', df['n_states'][i])

                print('\nTimescales:')
                print(df['timescales'][i])

                print('\nTransmat:')
                print(np.array(df['transmat'][i]))

                print('\nMeans:')
                print(np.array(df['means'][i]))

                if 'vars' in df:
                    print('\nVars:')
                    print(np.array(df['vars'][i]))
                elif 'kappas' in df:
                    print('\nKappas:')
                    print(np.array(df['kappas'][i]))

                print('\nTraining Logprobs (each EM iteration)')
                print(np.array(df['train_logprobs'][i]))

                if i < len(df)-1:
                    print('\n' + ('~'*70) + '\n')
