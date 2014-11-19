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

Written and maintained by TJ Lane <tjlane@stanford.edu>
Contributions from Kyle Beauchamp, Robert McGibbon, Vince Voelz,
Christian Schwantes.

These are the cannonical references for TPT. Note that TPT is really a
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

from msmbuilder import MSMLib
from msmbuilder import msm_analysis
from msmbuilder.utils import deprecated
import itertools
import copy

import logging
logger = logging.getLogger(__name__)

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)

######################################################################
# Functions for computing hub scores, conditional committors, and
# related quantities
#
def calculate_fraction_visits(tprob, waypoint, source, sink, return_cond_Q=False):
    """
    Calculate the fraction of times a walker on `tprob` going from `sources`
    to `sinks` will travel through the set of states `waypoints` en route.

    Computes the conditional committors q^{ABC^+} and uses them to find the
    fraction of paths mentioned above. The conditional committors can be

    Note that in the notation of Dickson et. al. this computes h_c(A,B), with
        sources   = A
        sinks     = B
        waypoint  = C

    Parameters
    ----------
    tprob : matrix
        The transition probability matrix
    waypoint : int
        The index of the intermediate state
    sources : nd_array, int or int
        The indices of the source state(s)
    sinks : nd_array, int or int
        The indices of the sink state(s)
    return_cond_Q : bool
        Whether or not to return the conditional committors

    Returns
    -------
    fraction_paths : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.
    cond_Q : nd_array, float (optional)
        Optionally returned (`return_cond_Q`)

    See Also
    --------
    calculate_hub_score : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.
    calculate_all_hub_scores : function
        Wrapper to compute all the hub scores in a network.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^3

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
          Article ASAP DOI: 10.1021/ct300537s
    """

    # do some typechecking - we need to be sure that the lumped sources are in
    # the second to last row, and the lumped sinks are in the last row
    # check `tprob`
    msm_analysis.check_transition(tprob)
    if type(tprob) != np.ndarray:
        try:
            tprob = tprob.todense()
        except AttributeError as e:
            raise TypeError('Argument `tprob` must be convertable to a dense'
                            'numpy array. \n%s' % e)

    # typecheck
    for data in [source, sink, waypoint]:
        if type(data) == int:
            pass
        elif hasattr(data, 'len'):
            if len(data) == 1:
                data = data[0]
        else:
            raise TypeError('Arguments source/sink/waypoint must be an int')

    if (source == waypoint) or (sink == waypoint) or (sink == source):
        raise ValueError('source, sink, waypoint must all be disjoint!')

    N = tprob.shape[0]
    Q = calculate_committors([source], [sink], tprob)

    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and source + sink to the end after that
    Bsink_indices = [source, sink, waypoint]
    perm = np.arange(N)
    perm = np.delete(perm, Bsink_indices)
    perm = np.append(perm, Bsink_indices)
    T = MSMLib.permute_mat(tprob, perm)

    # extract P, R
    n = N - len(Bsink_indices)
    P = T[:n, :n]
    R = T[:n, n:]

    # calculate the conditional committors ( B = N*R ), B[i,j] is the prob
    # state i ends in j, where j runs over the source + sink + waypoint
    # (waypoint is position -1)
    B = np.dot(np.linalg.inv(np.eye(n) - P), R)
    # Not sure if this is sparse or not...

    # add probs for the sinks, waypoint / b[i] is P( i --> {C & not A, B} )
    b = np.append(B[:, -1].flatten(), [0.0] * (len(Bsink_indices) - 1) + [1.0])
    cond_Q = b * Q[waypoint]

    epsilon = 1e-6  # some numerical give, hard-coded
    assert cond_Q.shape == (N,)
    assert np.all(cond_Q <= 1.0 + epsilon)
    assert np.all(cond_Q >= 0.0 - epsilon)
    assert np.all(cond_Q <= Q[perm] + epsilon)

    # finally, calculate the fraction of paths h_C(A,B) (eq. 7 in [1])
    fraction_paths = np.sum(T[-3, :] * cond_Q) / np.sum(T[-3, :] * Q[perm])

    assert fraction_paths <= 1.0
    assert fraction_paths >= 0.0

    if return_cond_Q:
        cond_Q = cond_Q[np.argsort(perm)]  # put back in orig. order
        return fraction_paths, cond_Q
    else:
        return fraction_paths


def calculate_hub_score(tprob, waypoint):
    """
    Calculate the hub score for the states `waypoint`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.


    Parameters
    ----------
    tprob : matrix
        The transition probability matrix
    waypoints : int
        The indices of the intermediate state(s)

    Returns
    -------
    Hc : float
        The hub score for the state composed of `waypoints`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".
    calculate_all_hub_scores : function
        A more efficient way to compute the hub score for every state in a
        network.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^5

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    msm_analysis.check_transition(tprob)

    # typecheck
    if type(waypoint) != int:
        if hasattr(waypoint, '__len__'):
            if len(waypoint) == 1:
                waypoint = waypoint[0]
            else:
                raise ValueError('Must pass waypoints as int or list/array of ints')
        else:
            raise ValueError('Must pass waypoints as int or list/array of ints')

    # find out which states to include in A, B (i.e. everything but C)
    N = tprob.shape[0]
    states_to_include = list(range(N))
    states_to_include.remove(waypoint)

    # calculate the hub score
    Hc = 0.0
    for s1 in states_to_include:
        for s2 in states_to_include:
            if (s1 != s2) and (s1 != waypoint) and (s2 != waypoint):
                Hc += calculate_fraction_visits(tprob, waypoint,
                                                s1, s2, return_cond_Q=False)

    Hc /= ((N - 1) * (N - 2))

    return Hc


def calculate_all_hub_scores(tprob):
    """
    Calculate the hub scores for all states in a network defined by `tprob`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.

    Parameters
    ----------
    tprob : matrix
        The transition probability matrix

    Returns
    -------
    Hc_array : nd_array, float
        The hub score for each state in `tprob`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".
    calculate_hub_score : function
        A function that computes just one hub score, can compute the hub score
        for a set of states.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^6

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    N = tprob.shape[0]
    states = list(range(N))

    # calculate the hub score
    Hc_array = np.zeros(N)

    # loop over each state and compute it's hub score
    for i, waypoint in enumerate(states):

        Hc = 0.0

        # now loop over all combinations of sources/sinks and average
        for s1 in states:
            if waypoint != s1:
                for s2 in states:
                    if s1 != s2:
                        if waypoint != s2:
                            Hc += calculate_fraction_visits(tprob, waypoint, s1, s2)

        # store the hub score in an array
        Hc_array[i] = Hc / ((N - 1) * (N - 2))

    return Hc_array
