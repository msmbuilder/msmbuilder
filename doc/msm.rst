.. _msm:
.. currentmodule:: mixtape.markovstatemodel

Markov state models (MSMs)
===========================

Background
----------

Markov state models (MSMs) are a class of timeseries models for modeling the long-timescale dynamics of molecular systems. An MSM is essentially a kinetic map of the conformational space a molecule explores. The model consists of (1) a set of conformational states and (2) a matrix of transition probabilities (or, equivalently, transition rates) between each pair of states. 

In mixtape, you can use :class:`MarkovStateModel` to build MSMs from "labeled" trajectories -- that is, sequences of integers corresponding to the index of the conformational state occupied by the system at each time point on a trajectory. The :ref:`cluster` module provides a number of different methods for
clustering the trajectories that you can use to define the states.


Algorithms
----------

.. autosummary::
    :toctree: generated/

    MarkovStateModel


TODO: Analyzing an MSM
----------------------
1. timescales
2. eigenvectors
3. lumping


References
----------
.. [#f1] Prinz, Jan-Hendrik, et al. `Markov models of molecular kinetics: Generation and validation <http://dx.doi.org/10.1063/1.3565032>`_ J. Chem. Phys. 134.17 (2011): 174105. 
.. [#f2] Pande, Vijay S., Kyle Beauchamp, and Gregory R. Bowman. `Everything you wanted to know about Markov State Models but were afraid to ask <http://dx.doi.org/10.1016/j.ymeth.2010.06.002>`_ Methods 52.1 (2010): 99-105.