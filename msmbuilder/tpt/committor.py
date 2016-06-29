# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""
Functions for computing forward committors for an MSM. The forward
committor is defined for a set of sources and sink states, and for
each state, the forward committor is the probability that a walker
starting at that state will visit the sink state before the source
state.

These are some canonical references for TPT. Note that TPT
is really a specialization of ideas very familiar to the
mathematical study of Markov chains, and there are many
books, manuscripts in the mathematical literature that
cover the same concepts.

References
----------
.. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
       transition paths. J. Stat. Phys. 123, 503-523 (2006).
.. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
       Transition path theory for Markov jump processes.
       Multiscale Model. Simul. 7, 1192-1219 (2009).
.. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
       flux and folding pathways in network models of
       coarse-grained protein dynamics. J. Chem. Phys.
       130, 205102 (2009).
.. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
       pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
       19011-19016.
"""
from __future__ import print_function, division, absolute_import
import numpy as np

from mdtraj.utils.six.moves import xrange

__all__ = ['committors', 'conditional_committors',
           '_committors', '_conditional_committors']


def committors(sources, sinks, msm):
    """
    Get the forward committors of the reaction sources -> sinks.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    msm : msmbuilder.MarkovStateModel
        MSM fit to the data.

    Returns
    -------
    forward_committors : np.ndarray
        The forward committors for the reaction sources -> sinks

    References
    ----------
    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
           transition paths. J. Stat. Phys. 123, 503-523 (2006).
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
           Transition path theory for Markov jump processes.
           Multiscale Model. Simul. 7, 1192-1219 (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
           flux and folding pathways in network models of
           coarse-grained protein dynamics. J. Chem. Phys.
           130, 205102 (2009).
    .. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
           19011-19016.
    """

    if hasattr(msm, 'all_transmats_'):
        commits = np.zeros(msm.all_transmats_.shape[:2])
        for i, tprob in enumerate(msm.all_transmats_):
            commits[i, :] = _committors(sources, sinks, tprob)
        return np.median(commits, axis=0)

    return _committors(sources, sinks, msm.transmat_)


def conditional_committors(source, sink, waypoint, msm):
    """
    Computes the conditional committors :math:`q^{ABC^+}` which are is the
    probability of starting in one state and visiting state B before A while
    also visiting state C at some point.

    Note that in the notation of Dickson et. al. this computes :math:`h_c(A,B)`,
    with ``sources = A``, ``sinks = B``, ``waypoint = C``

    Parameters
    ----------
    waypoint : int
        The index of the intermediate state
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    msm : msmbuilder.MarkovStateModel
        MSM to analyze.

    Returns
    -------
    cond_committors : np.ndarray
        Conditional committors, i.e. the probability of visiting
        a waypoint when on a path between source and sink.

    See Also
    --------
    msmbuilder.tpt.fraction_visited : function
        Calculate the fraction of visits to a waypoint from a given
        source to a sink.
    msmbuilder.tpt.hub_scores : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.

    Notes
    -----
    Employs dense linear algebra, memory use scales as N^2,
    and cycle use scales as N^3

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    # typecheck
    for data in [source, sink, waypoint]:
        if not isinstance(data, int):
            raise ValueError("source, sink, and waypoint must be integers.")

    if (source == waypoint) or (sink == waypoint) or (sink == source):
        raise ValueError('source, sink, waypoint must all be disjoint!')

    if hasattr(msm, 'all_transmats_'):
        cond_committors = np.zeros(msm.all_transmats_.shape[:2])
        for i, tprob in enumerate(msm.all_transmats_):
            cond_committors[i, :] = _conditional_committors(source, sink,
                                                            waypoint, tprob)
        return np.median(cond_committors, axis=0)

    return _conditional_committors(source, sink, waypoint, msm.transmat_)


def _conditional_committors(source, sink, waypoint, tprob):
    """
    Computes the conditional committors :math:`q^{ABC^+}` which are is the
    probability of starting in one state and visiting state B before A while
    also visiting state C at some point.

    Note that in the notation of Dickson et. al. this computes :math:`h_c(A,B)`,
    with ``sources = A``, ``sinks = B``, ``waypoint = C``

    Parameters
    ----------
    waypoint : int
        The index of the intermediate state
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    tprob : np.ndarray
        Transition matrix

    Returns
    -------
    cond_committors : np.ndarray
        Conditional committors, i.e. the probability of visiting
        a waypoint when on a path between source and sink.

    Notes
    -----
    Employs dense linear algebra, memory use scales as N^2,
    and cycle use scales as N^3

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    n_states = np.shape(tprob)[0]

    forward_committors = _committors([source], [sink], tprob)

    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and source + sink to the end after that
    Bsink_indices = [source, sink, waypoint]
    perm = np.array([i for i in xrange(n_states) if i not in Bsink_indices],
                    dtype=int)
    perm = np.concatenate([perm, Bsink_indices])
    permuted_tprob = tprob[perm, :][:, perm]

    # extract P, R
    n = n_states - len(Bsink_indices)
    P = permuted_tprob[:n, :n]
    R = permuted_tprob[:n, n:]

    # calculate the conditional committors ( B = N*R ), B[i,j] is the prob
    # state i ends in j, where j runs over the source + sink + waypoint
    # (waypoint is position -1)
    B = np.dot(np.linalg.inv(np.eye(n) - P), R)

    # add probs for the sinks, waypoint / b[i] is P( i --> {C & not A, B} )
    b = np.append(B[:, -1].flatten(), [0.0] * (len(Bsink_indices) - 1) + [1.0])
    cond_committors = b * forward_committors[waypoint]

    # get the original order
    cond_committors = cond_committors[np.argsort(perm)]

    return cond_committors


def _committors(sources, sinks, tprob):
    """
    Get the forward committors of the reaction sources -> sinks.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : np.ndarray
        Transition matrix

    Returns
    -------
    forward_committors : np.ndarray
        The forward committors for the reaction sources -> sinks

    References
    ----------
    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
           transition paths. J. Stat. Phys. 123, 503-523 (2006).
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
           Transition path theory for Markov jump processes.
           Multiscale Model. Simul. 7, 1192-1219 (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
           flux and folding pathways in network models of
           coarse-grained protein dynamics. J. Chem. Phys.
           130, 205102 (2009).
    .. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
           19011-19016.
    """
    n_states = np.shape(tprob)[0]

    sources = np.array(sources, dtype=int).reshape((-1, 1))
    sinks = np.array(sinks, dtype=int).reshape((-1, 1))

    # construct the committor problem
    lhs = np.eye(n_states) - tprob

    for a in sources:
        lhs[a, :] = 0.0  # np.zeros(n)
        lhs[:, a] = 0.0
        lhs[a, a] = 1.0

    for b in sinks:
        lhs[b, :] = 0.0  # np.zeros(n)
        lhs[:, b] = 0.0
        lhs[b, b] = 1.0

    ident_sinks = np.zeros(n_states)
    ident_sinks[sinks] = 1.0

    rhs = np.dot(tprob, ident_sinks)
    rhs[sources] = 0.0
    rhs[sinks] = 1.0

    forward_committors = np.linalg.solve(lhs, rhs)

    return forward_committors
