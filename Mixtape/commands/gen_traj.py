'''Generate a trajectory from a MSLDS Model'''
# Author: Bharath Ramsundar <rmcgibbo@gmail.com>
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

import sys
import numpy as np
import pandas as pd
from sklearn.mixture.gmm import log_multivariate_normal_density

from mixtape.cmdline import argument_group
from mixtape.commands.sample import SampleGHMM
import mixtape.featurizer

__all__ = ['PullMeansGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SampleMSLDS(Command, MDTrajInputMixin):
    name = 'sample-mslds'
    description = '''Sample a trajectory from a MSLDS model.'''

    group = argument_group('I/O Arguments')
    group.add_argument('-i', '--filename', required=True,
        metavar='JSONLINES_FILE', help=
        '''Path to the jsonlines output file containg the MSLDS''')
    group.add_argument('--featurizer', type=str, required=True,
        help='''Path to saved featurizer object''')
    group.add_argument('--n-states', type=int, required=True,
        help='''Number of states in the model to select from''')
    group.add_argument('--n-samples', type=int, required=True,
        help='''Length of trajectory to sample''')
    group.add_argument('--lag-time', type=int, required=True,
        help='''Training lag time of the model to select from''')
    group.add_argument('-o', '--out', metavar='OUTPUT_H5_FILE',
        default='traj.h5',
        help=('''File to which to save the output, ''' +
                '''in CSV format. default="means.csv''')

    def __init__(self, args):
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        matches = [o for o in iterobjects(args.filename)
                   if o['n_states'] == args.n_states]
        if len(matches) == 0:
            self.error('No model with n_states=%d, '
                + 'train_lag_time=%d in %s.' %
                (args.n_states, args.lag_time, args.filename))

        self.args = args
        self.model = matches[0]
        self.out = args.out
        self.topology = md.load(args.top)
        self.filenames = glob.glob(
            os.path.join(os.path.expanduser(args.dir), '*.%s' % args.ext))
        self.featurizer = mixtape.featurizer.load(args.featurizer)
        self.match_vars = args.match_vars

        if len(self.filenames) == 0:
            self.error('No files matched.')

    def start(self):
        featurizer = mixtape.featurizer.load(self.args.featurizer)
        obs, hidden_states = self.model.sample(args.n_samples)
        (n_samples, n_features) = shape(obs)

        features, ii, ff = mixtape.featurizer.featurize_all(
            self.filenames, featurizer, self.topology)


        states = []
        logprob = log_multivariate_normal_density(
            features, np.array(self.model['means']),
            np.array(self.model['covars']), covariance_type='diag')
        assignments = np.argmax(logprob, axis=1)
        probs = np.max(logprob, axis=1)
        # Presort the data into the metastable wells
        for k in range(self.model(['n_states'])):
            # pick the structures that have the highest log
            # probability in the state
            s = features[assignments == k]
            states.append(s)

        # Assign the best fit to each trajectory frame
        for t in range(n_samples):
            featurized_frame = obs[t]
            h = hidden_state[t]
            logprob = log_multivariate_normal_density(
                states[h], featurized_frame,
                np.array(self.model['Qs'][h]),
                covariance_type='full')
            best_frame_pos = np.argmax(logprob, axis=0)

        df = pd.DataFrame(data)
        with open(self.out, 'w') as f:
            f.write("# command: %s\n" % ' '.join(sys.argv))
            df.to_csv(f)
