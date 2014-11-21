# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
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
"""
Functions for performing mean first passage time calculations for an MSM.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import scipy
from mdtraj.utils.six.moves import xrange
import copy

__all__ = ['mfpts']

def mfpts(msm, sinks=None, lag_time=1.):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    msm : mixtape.MarkovStateModel
        MSM fit to the data.
    sinks : array, int
        Indices of the sink states. If None, then all MFPTs will be
        calculated using the Z-matrix formalism (which can be faster)
    lag_time : float, optional
        Lag time for the model. The MFPT will be reported in whatever
        units are given here. Default is (1) which is in units of the
        lag time of the MSM.

    Returns
    -------
    mfpts : np.ndarray, float
        MFPT in time units of lag_time, for each state (in order of state index)

    References
    ----------
    .. [1] E, Weinan and Vanden-Eijnden, Eric Towards a Theory of Transition Paths
           J. Stat. Phys. 123 503-523 (2006)
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding
           pathways in network models of coarse-grained protein dynamics. J.
           Chem. Phys. 130, 205102 (2009).
    """
    populations = msm.populations_
    tprob = msm.transmat_
    n_states = msm.n_states_

    if sinks is None:

        #eye = np.transpose(np.matrix(np.ones(num_states)))
        # ^^^^!!!!!!!^^^^^ who wrote this and thought it was acceptable?!
        # after plugging it into ipython, this just creates a column vector

        #limiting_matrix = eye * populations
        # since eye used to be a matrix, eye * populations was an
        # outer product with the populations in each of the columns
        # I'm pretty sure it's supposed to be in the rows, though...
        limiting_matrix = np.vstack([populations] * n_states)
        #z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))
        z = scipy.linalg.inv(np.eye(n_states) - (tprob - limiting_matrix))

        # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
        mfpts = - z
        for j in range(n_states):
            mfpts[:, j] += z[j, j]
            mfpts[:, j] /= populations[j]

        mfpts *= lag_time

    else:
        sinks = np.array(sinks, dtype=int).reshape((-1,))

        tprob = copy.copy(tprob)

        for state in sinks:
            tprob[state, :] = 0.0
            tprob[state, state] = 2.0

        tprob = tprob - np.eye(n_states)

        rhs = -1 * np.ones(n_states)
        for state in sinks:
            rhs[state] = 0.0

        mfpts = lag_time * np.linalg.solve(tprob, rhs)

    return mfpts
