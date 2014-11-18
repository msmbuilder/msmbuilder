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


def calculate_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Compute the transition path theory flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : mm_matrix
        The transition matrix.

    Returns
    ------
    fluxes : mm_matrix
        The flux matrix

    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True

    # check if we got the populations
    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 5)
        if np.count_nonzero(np.imag(eigens[1][:, 0])) != 0:
            raise ValueError('First eigenvector has imaginary components')
        populations = np.real(eigens[1][:, 0])

    # check if we got the committors
    if committors is None:
        committors = calculate_committors(sources, sinks, tprob)

    # perform the flux computation
    Indx, Indy = tprob.nonzero()

    n = tprob.shape[0]

    if dense:
        X = np.zeros((n, n))
        Y = np.zeros((n, n))
        X[(np.arange(n), np.arange(n))] = populations * (1.0 - committors)
        Y[(np.arange(n), np.arange(n))] = committors
    else:
        X = scipy.sparse.lil_matrix((n, n))
        Y = scipy.sparse.lil_matrix((n, n))
        X.setdiag(populations * (1.0 - committors))
        Y.setdiag(committors)

    if dense:
        fluxes = np.dot(np.dot(X, tprob), Y)
        fluxes[(np.arange(n), np.arange(n))] = np.zeros(n)
    else:
        fluxes = (X.tocsr().dot(tprob.tocsr())).dot(Y.tocsr())
        # This should be the same as below, but it's a bit messy...
        #fluxes = np.dot(np.dot(X.tocsr(), tprob.tocsr()), Y.tocsr())
        fluxes = fluxes.tolil()
        fluxes.setdiag(np.zeros(n))

    return fluxes


def calculate_net_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Computes the transition path theory net flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : mm_matrix
        The transition matrix.

    Returns
    ------
    net_fluxes : mm_matrix
        The net flux matrix

    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True

    n = tprob.shape[0]

    flux = calculate_fluxes(sources, sinks, tprob, populations, committors)
    ind = flux.nonzero()

    if dense:
        net_flux = np.zeros((n, n))
    else:
        net_flux = scipy.sparse.lil_matrix((n, n))

    for k in range(len(ind[0])):
        i, j = ind[0][k], ind[1][k]
        forward = flux[i, j]
        reverse = flux[j, i]
        net_flux[i, j] = max(0, forward - reverse)

    return net_flux


