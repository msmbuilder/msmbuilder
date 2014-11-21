.. _tpt:
.. currentmodule:: mixtape.tpt

Transition Path Theory
======================


Background
----------
This module contains functions for analyzing Markov state models, with an
emphasis on Transition Path Theory (TPT)

These are the canonical references for TPT. Note that TPT is really a
specialization of ideas very framiliar to the mathematical study of Markov
chains, and there are many books, manuscripts in the mathematical literature
that cover the same concepts.

References
~~~~~~~~~~
.. [1] E, Weinan and Vanden-Eijnden, Eric Towards a Theory of Transition Paths
       J. Stat. Phys. 123 503-523 (2006)
.. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
       for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
       (2009).
.. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
       pathways in network models of coarse-grained protein dynamics. J. 
       Chem. Phys. 130, 205102 (2009).


Functions
---------

.. autosummary::
    :toctree: generated/

    fluxes
    net_fluxes
    fraction_visited
    hub_scores
    paths
    top_path
    committors
    conditional_committors
    mfpts
