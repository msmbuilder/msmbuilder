# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""
Functions for analyzing the "hub-ness" of an MSM.

References
----------
.. [1] Dickson, Alex, and Charles L. Brooks III. "Quantifying
       hub-like behavior in protein folding networks."
       JCTC, 8, 3044-3052 (2012).
"""
from __future__ import print_function, division, absolute_import
import numpy as np

from . import committors, conditional_committors

from mdtraj.utils.six.moves import xrange
import itertools

__all__ = ['fraction_visited', 'hub_scores']


def fraction_visited(source, sink, waypoint, msm):
    """
    Calculate the fraction of times a walker on `tprob` going from `sources`
    to `sinks` will travel through the set of states `waypoints` en route.

    Computes the conditional committors q^{ABC^+} and uses them to find the
    fraction of paths mentioned above.

    Note that in the notation of Dickson et. al. this computes h_c(A,B), with
        sources   = A
        sinks     = B
        waypoint  = C

    Parameters
    ----------
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    waypoint : int
        The index of the intermediate state
    msm : msmbuilder.MarkovStateModel
        MSM to analyze.

    Returns
    -------
    fraction_visited : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.

    See Also
    --------
    msmbuilder.tpt.conditional_committors
        Calculate the probability of visiting a waypoint while on a path
        between a source and sink.
    msmbuilder.tpt.hub_scores : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    for_committors = committors([source], [sink], msm)
    cond_committors = conditional_committors(source, sink, waypoint, msm)

    if hasattr(msm, 'all_transmats_'):
        frac_visited = np.zeros((msm.n_states,))
        for i, tprob in enumerate(msm.all_transmats_):
            frac_visited[i] = _fraction_visited(source, sink, waypoint,
                                                msm.transmat_, for_committors,
                                                cond_committors)
        return np.median(frac_visited, axis=0)

    return _fraction_visited(source, sink, waypoint, msm.transmat_,
                             for_committors, cond_committors)


def hub_scores(msm, waypoints=None):
    """
    Calculate the hub score for one or more waypoints

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.


    Parameters
    ----------
    msm : msmbuilder.MarkovStateModel
        MSM to analyze
    waypoints : array_like, int, optional
        The index of the intermediate state (or more than one).
        If None, then all waypoints will be used

    Returns
    -------
    hub_score : float
        The hub score for the waypoint

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    n_states = msm.n_states_
    if isinstance(waypoints, int):
        waypoints = [waypoints]
    elif waypoints is None:
        waypoints = xrange(n_states)
    elif not (isinstance(waypoints, list) or
              isinstance(waypoints, np.ndarray)):
        raise ValueError("waypoints (%s) must be an int, a list, or None" %
                         str(waypoints))

    hub_scores = []
    for waypoint in waypoints:
        other_states = (i for i in xrange(n_states) if i != waypoint)

        # calculate the hub score for this waypoint
        hub_score = 0.0
        for (source, sink) in itertools.permutations(other_states, 2):
            hub_score += fraction_visited(source, sink, waypoint, msm)

        hub_score /= float((n_states - 1) * (n_states - 2))
        hub_scores.append(hub_score)

    return np.array(hub_scores)


def _fraction_visited(source, sink, waypoint, tprob, for_committors,
                      cond_committors):
    """
    Calculate the fraction of times a walker on `tprob` going from `sources`
    to `sinks` will travel through the set of states `waypoints` en route.

    Computes the conditional committors q^{ABC^+} and uses them to find the
    fraction of paths mentioned above.

    Note that in the notation of Dickson et. al. this computes h_c(A,B), with
        sources   = A
        sinks     = B
        waypoint  = C

    Parameters
    ----------
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    waypoint : int
        The index of the intermediate state
    tprob : np.ndarray
        Transition matrix
    for_committors : np.ndarray
        The forward committors for the reaction sources -> sinks
    cond_committors : np.ndarray
        Conditional committors, i.e. the probability of visiting
        a waypoint when on a path between source and sink.

    Returns
    -------
    fraction_visited : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.

    See Also
    --------
    msmbuilder.tpt.conditional_committors
        Calculate the probability of visiting a waypoint while on a path
        between a source and sink.
    msmbuilder.tpt.hub_scores : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    fraction_visited = (np.float(tprob[source, :].dot(cond_committors)) /
                        np.float(tprob[source, :].dot(for_committors)))

    return fraction_visited
