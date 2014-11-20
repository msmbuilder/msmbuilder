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
Functions for analyzing the "hub-ness" of an MSM.

References
----------
.. [1] Dickson, Alex, and Charles L. Brooks III. "Quantifying 
       hub-like behavior in protein folding networks." 
       JCTC, 8, 3044-3052 (2012).
"""
from __future__ import print_function, division, absolute_import
import numpy as np

from . import committors

import itertools
import copy

__all__ = ['fraction_visited', 'hub_scores']

def fraction_visited(source, sink, waypoint, msm):
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
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    waypoint : int
        The index of the intermediate state
    msm : mixtape.MarkovStateModel
        MSM to analyze.

    Returns
    -------
    fraction_visited : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.

    See Also
    --------
    committors.calculate_conditional_committors
        Calculate the probability of visiting a waypoint while on a path
        between a source and sink.
    hubs.calculate_hub_score : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.
    hubs.calculate_all_hub_scores : function
        Wrapper to compute all the hub scores in a network.

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    tprob = msm.transmat_
    n_states = msm.n_states_

    # EFFICIENCY ALERT:
    # we could allow all of these functions to pass committors if they've
    # already been calculated, but I think it's so fast that we don't need to
    committors = tpt.committors.calculate_committors([source], [sink], msm)
    cond_committors = tpt.committors.calculate_conditional_committors([source], [sink], msm)

    # OLD CODE: The transition matrix is permuted so that the last three are source, sink, waypoint
    #fraction_paths = np.sum(permuted_tprob[-3, :] * cond_committors) / np.sum(permuted_tprob[-3, :] * committors[perm])

    fraction_visited = np.float(tprob[source, :].dot(cond_committors)) / np.float(tprob[source, :].dot(committors))

    return fraction_visited


def hub_scores(msm, waypoints=None):
    """
    Calculate the hub score for one or more waypoints

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.


    Parameters
    ----------
    msm : mixtape.MarkovStateModel
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
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    tprob = msm.transmat_
    n_states = msm.n_states_
    if isinstance(waypoints, int):
        waypoints = [waypoints]
    elif waypoints is None:
        waypoints = xrange(n_states)
    elif not (isinstance(waypoints, list) or isinstance(waypoints, np.ndarray)):
        raise ValueError("waypoint (%s) must be an int a list or None" % str(waypoint))

    hub_scores = []
    for waypoint in waypoints:
        other_states = (i for i in xrange(n_states) if i != waypoint)

        # calculate the hub score for this waypoint
        hub_score = 0.0
        for (source, sink) in itertools.permutations(other_states, 2):
            hub_score += calculate_fraction_visits(source, sink, waypoint, msm)

        hub_score /= float((N - 1) * (N - 2))
        hub_scores.append(hub_score)

    return np.array(hub_scores)
