.. _tpt:
.. currentmodule:: msmbuilder.tpt

Transition Path Theory
======================


Transition path theory (TPT) is a way to extract the highest-flux pathways
of your system from an estimated MSM. 

.. todo: more

.. todo: example


References
----------

These are some canonical references for TPT. Note that TPT is really a
specialization of ideas very familiar to the mathematical study of Markov
chains, and there are many books, manuscripts in the mathematical
literature that cover the same concepts.

.. [1] E, Weinan and Vanden-Eijnden, Eric Towards a Theory of Transition Paths
       J. Stat. Phys. 123 503-523 (2006)
.. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory
       for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
       (2009).
.. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding
       pathways in network models of coarse-grained protein dynamics. J.
       Chem. Phys. 130, 205102 (2009).
.. [4] Noé, Frank, et al. "Constructing the equilibrium ensemble of folding
       pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
       19011-19016.

Functions
---------

.. autosummary::
    :toctree: _tpt/

    fluxes
    net_fluxes
    fraction_visited
    hub_scores
    paths
    top_path
    committors
    conditional_committors
    mfpts

.. vim: tw=75
