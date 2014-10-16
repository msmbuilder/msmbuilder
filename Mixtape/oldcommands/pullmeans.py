'''Get structures at the center of each state in an HMM
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

import sys
import numpy as np
import pandas as pd
from sklearn.mixture.gmm import log_multivariate_normal_density

from mixtape.cmdline import argument_group
from mixtape.oldcommands.sample import SampleGHMM
import mixtape.featurizer

__all__ = ['PullMeansGHMM']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class PullMeansGHMM(SampleGHMM):
    _concrete = True
    name = 'means-ghmm'
    description = '''Draw samples at the center of each state in a Gaussian HMM.'''

    group = argument_group('I/O Arguments')
    group.add_argument('-i', '--filename', required=True, metavar='JSONLINES_FILE', help='''
        Path to the jsonlines output file containg the HMMs''')
    group.add_argument('--featurizer', type=str, required=True, help='''
        Path to saved featurizer object''')
    group.add_argument('--stride', type=int, default=1, help='''
        Load up only every stride-th frame from the trajectories, to reduce
        memory usage''')
    group.add_argument('--n-states', type=int, required=True, help='''Number of
        states in the model to select from''')
    group.add_argument('--n-per-state', type=int, default=1, help='''Select the
        `n-per-state` most representative structures from each state. default=1''')
    group.add_argument('--lag-time', type=int, required=True, help='''Training lag
        time of the model to select from''')
    group.add_argument('-o', '--out', metavar='OUTPUT_CSV_FILE', default='means.csv', help='''
        File to which to save the output, in CSV format. default="means.csv''')

    def start(self):
        featurizer = mixtape.featurizer.load(self.args.featurizer)

        features, ii, ff = mixtape.featurizer.featurize_all(
            self.filenames, featurizer, self.topology, self.stride)
        logprob = log_multivariate_normal_density(
            features, np.array(self.model['means']),
            np.array(self.model['vars']), covariance_type='diag')

        assignments = np.argmax(logprob, axis=1)
        probs = np.max(logprob, axis=1)

        data = {'filename': [], 'index': [], 'state': []}
        for k in range(self.model['n_states']):
            # pick the structures that have the highest log
            # probability in the state
            p = probs[assignments == k]
            sorted_filenms = ff[assignments == k][p.argsort()]
            sorted_indices = ii[assignments == k][p.argsort()]

            if len(p) > 0:
                data['index'].extend(sorted_indices[-self.args.n_per_state:])
                index_length = len(sorted_indices[-self.args.n_per_state:])
                data['filename'].extend(sorted_filenms[-self.args.n_per_state:])
                filename_length = len(sorted_filenms[-self.args.n_per_state:])
                assert index_length == filename_length
                data['state'].extend([k] * index_length)
            else:
                print('WARNING: NO STRUCTURES ASSIGNED TO STATE=%d' % k)

        df = pd.DataFrame(data)
        print('Saving the indices of the selected frames in CSV format to %s' % self.out)
        with open(self.out, 'w') as f:
            f.write("# command: %s\n" % ' '.join(sys.argv))
            df.to_csv(f)
