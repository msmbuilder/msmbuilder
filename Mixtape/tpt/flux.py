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
Functions for calculating the fluxes through an MSM for a given set
of source and sink states.

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

from . import committors

import itertools
import copy

__all__ = ['fluxes', 'net_fluxes']

def fluxes(sources, sinks, msm, committors=None):
    """
    Compute the transition path theory flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    msm : mixtape.MarkovStateModel
        MSM that has been fit to data. 
    committors : np.ndarray, optional
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    Returns
    ------
    flux_matrix : np.ndarray
        The flux matrix

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """
    sources = np.array(sources).reshape((-1,))
    sinks = np.array(sinks).reshape((-1,))

    populations = msm.populations_
    tprob = msm.transmat_
    n_states = msm.n_states_

    # check if we got the committors
    if committors is None:
        committors = committors.calculate_committors(sources, sinks, tprob)
    else:
        committors = np.array(committors)
        if committors.shape != (n_states,):
            raise ValueError("Shape of committors %s should be %s" % (str(committors.shape), str((n_states,))))

    X = np.zeros((n_states, n_states))
    X[(np.arange(n_states), np.arange(n_states))] = populations * (1.0 - committors)

    Y = np.zeros((n_states, n_states))
    Y[(np.arange(n_states), np.arange(n_states))] = committors

    fluxes = np.dot(np.dot(X, tprob), Y)
    fluxes[(np.arange(n_states), np.arange(n_states))] = np.zeros(n)

    return fluxes


def net_fluxes(sources, sinks, msm, committors=None):
    """
    Computes the transition path theory net flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    msm : mixtape.MarkovStateModel
        MSM fit to data.
    committors : np.ndarray, optional
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    Returns
    ------
    net_flux : np.ndarray
        The net flux matrix

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    flux_matrix = calculate_fluxes(sources, sinks, msm, committors=committors)

    net_flux = flux_matrix - flux_matrix.T
    net_flux[np.where(net_flux < 0)] = 0.0

    # Old Code:
    #for k in range(len(ind[0])):
    #    i, j = ind[0][k], ind[1][k]
    #    forward = flux[i, j]
    #    reverse = flux[j, i]
    #    net_flux[i, j] = max(0, forward - reverse)

    return net_flux
