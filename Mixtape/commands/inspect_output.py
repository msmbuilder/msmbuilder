from __future__ import print_function, division
import os
import sys
import numpy as np
import pandas as pd
import mdtraj as md
from mixtape.utils import iterobjects
from mixtape.cmdline import Command, argument, argument_group

__all__ = ['Inspect']

class Inspect(Command):
    description = "Inspect the content of a '.jsonlines' output file"
    input = argument('-i', '--input', required=True,
        help='Path to .jsonlines file')

    details = argument('--details', action='store_true', help='Print all of the details of each model to stdout')
    n_states = argument('--n-states', type=int, help='Filter the output, only showing models with the specified n_states')
    lag_time = argument('--lag-time', type=int, help='Filter the output, only showing models with the specified lag_time')

    def __init__(self, args):
        self.args = args
        self.models = list(iterobjects(self.args.input))
        self.df = pd.DataFrame(self.models)

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
        print('\n')

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
                print('\nVars:')
                print(np.array(df['vars'][i]))
                print('\nTraining Logprobs (each EM iteration)')
                print(np.array(df['train_logprobs'][i]))
                print('\n\n')
