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
Functions for performing Transition Path Theory calculations. 

Contributions from Kyle Beauchamp, Robert McGibbon, Vince Voelz,
Christian Schwantes, and TJ Lane.

These are the canonical references for TPT. Note that TPT is really a
specialization of ideas very framiliar to the mathematical study of Markov
chains, and there are many books, manuscripts in the mathematical literature
that cover the same concepts.

References
----------
.. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory
       for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
       (2009).
.. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
       pathways in network models of coarse-grained protein dynamics. J. 
       Chem. Phys. 130, 205102 (2009).
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.sparse

import itertools
import copy

import logging
logger = logging.getLogger(__name__)

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)

###############################################################################
# Typechecking/Utility Functions
#


def _ensure_iterable(arg):
    if not hasattr(arg, '__iter__'):
        arg = list([int(arg)])
        logger.debug("Passed object was not iterable,"
                     " converted it to: %s" % str(arg))
    assert hasattr(arg, '__iter__')
    return arg

def _check_sources_sinks(sources, sinks):
    sources = _ensure_iterable(sources)
    sinks = _ensure_iterable(sinks)

    for s in sources:
        if s in sinks:
            raise ValueError("sources and sinks are not disjoint")

    return sources, sinks


###############################################################################
# Path Finding Functions
#

def calculate_mfpt(sinks, tprob, lag_time=1.):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    sinks : array, int
        indices of the sink states
    tprob : matrix
        transition probability matrix
    LagTime : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, for each state (in order of state index)

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

    sinks = _ensure_iterable(sinks)
    msm_analysis.check_transition(tprob)

    n = tprob.shape[0]

    if scipy.sparse.isspmatrix(tprob):
        tprob = tprob.tolil()

    for state in sinks:
        tprob[state, :] = 0.0
        tprob[state, state] = 2.0

    if scipy.sparse.isspmatrix(tprob):
        tprob = tprob - scipy.sparse.eye(n, n)
        tprob = tprob.tocsr()
    else:
        tprob = tprob - np.eye(n)

    RHS = -1 * np.ones(n)
    for state in sinks:
        RHS[state] = 0.0

    if scipy.sparse.isspmatrix(tprob):
        MFPT = lag_time * scipy.sparse.linalg.spsolve(tprob, RHS)
    else:
        MFPT = lag_time * np.linalg.solve(tprob, RHS)

    return MFPT


def calculate_all_to_all_mfpt(tprob, populations=None):
    """
    Calculate the all-states by all-state matrix of mean first passage
    times.

    This uses the fundamental matrix formalism, and should be much faster
    than GetMFPT for calculating many MFPTs.

    Parameters
    ----------
    tprob : matrix
        transition probability matrix
    populations : array_like, float
        optional argument, the populations of each state. If  not supplied,
        it will be computed from scratch

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, square array for MFPT from i -> j

    See Also
    --------
    GetMFPT : function
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

    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        tprob = tprob.toarray()
        logger.warning('calculate_all_to_all_mfpt does not support sparse linear algebra')

    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 1)
        if np.count_nonzero(np.imag(eigens[1][:, 0])) != 0:
            raise ValueError('First eigenvector has imaginary parts')
        populations = np.real(eigens[1][:, 0])

    # ensure that tprob is a transition matrix
    msm_analysis.check_transition(tprob)
    num_states = len(populations)
    if tprob.shape[0] != num_states:
        raise ValueError("Shape of tprob and populations vector don't match")

    eye = np.transpose(np.matrix(np.ones(num_states)))
    limiting_matrix = eye * populations
    #z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))
    z = scipy.linalg.inv(np.eye(num_states) - (tprob - limiting_matrix))

    # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
    mfpt = -z
    for j in range(num_states):
        mfpt[:, j] += z[j, j]
        mfpt[:, j] /= populations[j]

    return mfpt

