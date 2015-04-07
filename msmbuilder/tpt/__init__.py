"""
Module for analyzing Markov State Models, with an emphasis
on Transition Path Theory.

These are the canonical references for TPT. Note that TPT
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

from __future__ import absolute_import

from .committor import committors, conditional_committors
from .flux import fluxes, net_fluxes
from .hub import fraction_visited, hub_scores
from .path import paths, top_path
from .mfpt import mfpts

__all__ = ['fluxes', 'net_fluxes', 'fraction_visited',
           'hub_scores', 'paths', 'top_path', 'committors',
           'conditional_committors', 'mfpts']
