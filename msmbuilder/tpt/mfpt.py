# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""
Functions for performing mean first passage time
calculations for an MSM.

For a useful introduction to Markov Chains (both ergodic
and absorbing) check out Chapter 11 in:

.. [1] Grinstead, C. M. and Snell, J. L. Introduction to
       Probability. American Mathematical Soc., 1998.

The absorbing Markov Chain information is interesting, but
note that we are using ergodic Markov Chains.
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
    msm : msmbuilder.MarkovStateModel
        MSM fit to the data.
    sinks : array_like, int, optional
        Indices of the sink states. There are two use-cases:
            - None [default] : All MFPTs will be calculated, and the
                result is a matrix of the MFPT from state i to state j.
                This uses the fundamental matrix formalism.
            - list of ints or int : Only the MFPTs into these sink
                states will be computed. The result is a vector, with
                entry i corresponding to the average time it takes to
                first get to *any* sink state from state i
    lag_time : float, optional
        Lag time for the model. The MFPT will be reported in whatever
        units are given here. Default is (1) which is in units of the
        lag time of the MSM.

    Returns
    -------
    mfpts : np.ndarray, float
        MFPT in time units of lag_time, which depends on the input
        value of sinks:

        - If sinks is None, then mfpts's shape is (n_states, n_states).
            Where mfpts[i, j] is the mean first passage time to state j
            from state i.

        - If sinks contains one or more states, then mfpts's shape
            is (n_states,). Where mfpts[i] is the mean first passage
            time from state i to any state in sinks.

    References
    ----------
    .. [1] Grinstead, C. M. and Snell, J. L. Introduction to
           Probability. American Mathematical Soc., 1998.

    As of November 2014, this chapter was available for free online:
        http://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """

    n_states = msm.n_states_
    if hasattr(msm, 'all_transmats_'):
        tprob = msm.all_transmats_.mean(axis=0)
        populations = msm.all_populations_.mean(axis=0)
    else:
        tprob = msm.transmat_
        populations = msm.populations_

    if sinks is None:
        # Use Thm 11.16 in [1]
        limiting_matrix = np.vstack([populations] * n_states)

        # Fundamental matrix
        fund_matrix = scipy.linalg.inv(np.eye(n_states) - tprob + limiting_matrix)

        # mfpt[i,j] = (fund_matrix[j,j] - fund_matrix[i,j]) / populations[j]
        mfpts = fund_matrix * -1
        for j in xrange(n_states):
            mfpts[:, j] += fund_matrix[j, j]
            mfpts[:, j] /= populations[j]

        mfpts *= lag_time

    else:
        # See section 11.5, and use Thm 11.5
        # Turn our ergodic MSM into an absorbing one (all sink
        # states are absorbing). Then calculate the mean time
        # to absorption.
        # Note: we are slightly modifying the description in
        # 11.5 so that we also get the mfpts[sink] = 0.0
        sinks = np.array(sinks, dtype=int).reshape((-1,))

        absorb_tprob = copy.copy(tprob)

        for state in sinks:
            absorb_tprob[state, :] = 0.0
            absorb_tprob[state, state] = 2.0
            # note it has to be 2 because we subtract
            # the identity below.

        lhs = np.eye(n_states) - absorb_tprob

        rhs = np.ones(n_states)
        for state in sinks:
            rhs[state] = 0.0

        mfpts = lag_time * np.linalg.solve(lhs, rhs)

    return mfpts
