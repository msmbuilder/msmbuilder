"""Utility functions"""
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

import json
import numpy as np
from sklearn.utils import check_random_state
from numpy.linalg import norm

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass


def categorical(pvals, size=None, random_state=None):
    """Return random integer from a categorical distribution

    Parameters
    ----------
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1.
    size : int or tuple of ints, optional
        Defines the shape of the returned array of random integers. If None
        (the default), returns a single float.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    """
    cumsum = np.cumsum(pvals)
    if size is None:
        size = (1,)
        axis = 0
    elif isinstance(size, tuple):
        size = size + (1,)
        axis = len(size) - 1
    else:
        raise TypeError('size must be an int or tuple of ints')

    random_state = check_random_state(random_state)
    return np.sum(cumsum < random_state.random_sample(size), axis=axis)


##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V

# TODO: FIX THIS!
def compute_eigenspectra(self):
    """
    Compute the eigenspectra of operators A_i
    """
    eigenspectra = np.zeros((self.n_states,
                            self.n_features, self.n_features))
    for k in range(self.n_states):
        eigenspectra[k] = np.diag(np.linalg.eigvals(self.As_[k]))
    return eigenspectra

# TODO: FIX THIS!
def load_from_json_dict(model, model_dict):
    # Check that the num of states and features agrees
    n_features = float(model_dict['n_features'])
    n_states = float(model_dict['n_states'])
    if n_features != self.n_features or n_states != self.n_states:
        raise ValueError('Invalid number of states or features')
    # read array values from the json dictionary
    Qs = []
    for Q in model_dict['Qs']:
        Qs.append(np.array(Q))
    As = []
    for A in model_dict['As']:
        As.append(np.array(A))
    bs = []
    for b in model_dict['bs']:
        bs.append(np.array(b))
    means = []
    for mean in model_dict['means']:
        means.append(np.array(mean))
    covars = []
    for covar in model_dict['covars']:
        covars.append(np.array(covar))
    # Transmat
    transmat = np.array(model_dict['transmat'])
    # Create the MSLDS model
    self.Qs_ = Qs
    self.As_ = As
    self.bs_ = bs
    self.means_ = means
    self.covars_ = covars
    self.transmat_ = transmat
