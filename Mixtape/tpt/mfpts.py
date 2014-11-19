# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
Functions for performing mean first passage time calculations for an MSM.

Contributions from Kyle Beauchamp, Robert McGibbon, Vince Voelz,
Christian Schwantes, and TJ Lane.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import copy

from mixtape.tpt import calculate_committors

import logging
logger = logging.getLogger(__name__)

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)

def calculate_mfpts(sinks, msm, lag_time=1.):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    sinks : array, int
        Indices of the sink states
    msm : mixtape.MarkovStateModel
        MSM fit to the data.
    lag_time : float, optional
        Lag time for the model. The MFPT will be reported in whatever
        units are given here. Default is (1) which is in units of the
        lag time of the MSM.

    Returns
    -------
    mfpts : np.ndarray, float
        MFPT in time units of lag_time, for each state (in order of state index)

    See Also
    --------
    calculate_all_to_all_mfpt : function
        A more efficient way to calculate all the MFPTs in a network

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sinks = np.array(sinks, dtype=int).reshape((-1,))

    tprob = copy.copy(msm.transmat_)
    n_states = msm.n_states_

    for state in sinks:
        tprob[state, :] = 0.0
        tprob[state, state] = 2.0

    tprob = tprob - np.eye(n_states)

    rhs = -1 * np.ones(n_states)
    for state in sinks:
        rhs[state] = 0.0

    mfpts = lag_time * np.linalg.solve(tprob, rhs)

    return mfpts


def calculate_all_mfpts(msm, lag_time=1.0):
    """
    Calculate the all-states by all-state matrix of mean first passage
    times.

    This uses the fundamental matrix formalism, and should be much faster
    than "calculate_mfpts" for calculating many MFPTs.

    Parameters
    ----------
    msm : mixtape.MarkovStateModel
        MSM fit to the data
    lag_time : float, optional
        Lag time of the model. The units of the MFPTs will be in
        whatever units specified here. By default, lag_time is equal
        to one, which corresponds to units of MSM lag times.

    Returns
    -------
    mfpts : array, float
        MFPTs in time units of lag_time, square array for MFPT from i -> j

    See Also
    --------
    calculate_mfpts : function
        for calculating a subset of the MFPTs, with functionality for including
        a set of sinks

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    populations = msm.populations_
    tprob = msm.tprob_
    n_states = msm.n_states_

    #eye = np.transpose(np.matrix(np.ones(num_states)))
    # ^^^^!!!!!!!^^^^^ who wrote this and thought it was acceptable?!
    # after plugging it into ipython, this just creates a column vector
    eye = np.ones(num_states).reshape((-1, 1))

    limiting_matrix = eye * populations
    #z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))
    z = scipy.linalg.inv(np.eye(num_states) - (tprob - limiting_matrix))

    # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
    mfpt = - z
    for j in range(num_states):
        mfpt[:, j] += z[j, j]
        mfpt[:, j] /= populations[j]

    return mfpt

