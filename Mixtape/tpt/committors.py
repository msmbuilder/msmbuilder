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

import itertools
import copy

import logging
logger = logging.getLogger(__name__)

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)

def calculate_committors(sources, sinks, msm):
    """
    Get the forward committors of the reaction sources -> sinks.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    msm : mixtape.MarkovStateModel
        MSM fit to the data.

    Returns
    -------
    committors : np.ndarray
        The forward committors for the reaction sources -> sinks

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources = np.array(sources, dtype=int).reshape((-1, 1))
    sinks = np.array(sinks, dtype=int).reshape((-1, 1))

    tprob = msm.transmat_
    # construct the committor problem
    n_states = tprob.shape[0]

    T = np.eye(n_states) - tprob

    for a in sources:
        T[a, :] = 0.0  # np.zeros(n)
        T[:, a] = 0.0
        T[a, a] = 1.0

    for b in sinks:
        T[b, :] = 0.0  # np.zeros(n)
        T[:, b] = 0.0
        T[b, b] = 1.0

    IdB = np.zeros(n_states)
    IdB[sinks] = 1.0

    RHS = np.dot(tprob, IdB)
    RHS[sources] = 0.0
    RHS[sinks] = 1.0

    committors = np.linalg.solve(T, RHS)

    # we can probably (?) remove these assersion lines
    epsilon = 0.001
    assert np.all(committors <= (1.0 + epsilon))
    assert np.all(committors >= (0.0 - epsilon))

    return committors


