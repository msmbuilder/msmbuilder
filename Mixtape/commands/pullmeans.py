'''Get structures at the center of each state in an HMM
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

import numpy as np
import pandas as pd
import mdtraj as md
from sklearn.mixture.gmm import log_multivariate_normal_density

from mixtape.cmdline import argument_group
from mixtape.commands.sample import SampleGHMM
import mixtape.featurizer

__all__ = ['PullMeansGHMM']

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

class PullMeansGHMM(SampleGHMM):
    name='means-ghmm'
    description='''Draw samples at the center of each state in a Gaussian HMM.

    '''
    nps = None  # override this from superclass

    def start(self):
        featurizer = mixtape.featurizer.load(self.args.featurizer)
        logprob = []
        ff = []
        ii = []
        for file in self.filenames:
            kwargs = {}  if file.endswith('.h5') else {'top': topology}
            t = md.load(file, **kwargs)
            features = featurizer.featurize(t)
            logprob_local = log_multivariate_normal_density(features, np.array(self.model['means']),
                np.array(self.model['vars']), covariance_type='diag')
            logprob.extend(logprob_local)
            ii.append(np.arange(len(features)))
            ff.extend([file]*len(features))
        
        ii = np.concatenate(ii)
        ff = np.array(ff)
        
        logprob = np.array(logprob)
        assignments = np.argmax(logprob, axis=1)
        probs = np.max(logprob, axis=1)

        data = {'filename': [], 'index': [], 'state': []}
        for k in range(self.model['n_states']):
            # pick the structures that have the highest log probability in the state
            p = probs[assignments==k]
            sorted_filenms = ff[assignments==k][p.argsort()]
            sorted_indices = ii[assignments==k][p.argsort()]

            data['index'].append(sorted_indices[-1])
            data['filename'].append(sorted_filenms[-1])
            data['state'].append(k)

        df = pd.DataFrame(data)
        print('Saving the indices of the sampled states in CSV format to %s' % self.out)
        df.to_csv(self.out)
