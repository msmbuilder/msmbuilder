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
def calculate_fraction_visits(source, sink, waypoint, msm):
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
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
          Article ASAP DOI: 10.1021/ct300537s
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


def calculate_hub_score(msm, waypoint):
    """
    Calculate the hub score for a single `waypoint`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.


    Parameters
    ----------
    msm : mixtape.MarkovStateModel
        MSM to analyze
    waypoint : int
        The index of the intermediate state

    Returns
    -------
    hub_score : float
        The hub score for the waypoint

    See Also
    --------
    hubs.calculate_all_hub_scores : function
        A more efficient way to compute the hub score for every state in a
        network.

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    tprob = msm.transmat_
    n_states = msm.n_states_
    if not isinstance(waypoint, int):
        raise ValueError("waypoint (%s) must be an int" % str(waypoint))

    other_states = (i for i in xrange(n_states) if i != waypoint)

    # calculate the hub score
    hub_score = 0.0
    for (source, sink) in itertools.permutations(other_states, 2):
        hub_score += calculate_fraction_visits(source, sink, waypoint, msm)

    hub_score /= float((N - 1) * (N - 2))

    return hub_score


def calculate_all_hub_scores(msm):
    """
    Calculate the hub scores for all states in a network defined by `msm`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.

    Parameters
    ----------
    msm : mixtape.MarkovStateModel
        MSM to analyze.

    Returns
    -------
    hub_scores : np.ndarray
        The hub score for each state in `msm`

    See Also
    --------
    hubs.calculate_hub_score : function
        A function that computes just one hub score, can compute the hub score
        for a set of states.

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    n_states = msm.n_states_

    hub_scores = np.array([calculate_hub_score(msm, state) for state in xrange(n_states)])

    return hub_scores
