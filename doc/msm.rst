.. _msm:
.. currentmodule:: mixtape.markovstatemodel

Markov state models (MSMs)
===========================

Background
----------
Markov state models (MSMs) are a class of timeseries models for modeling the
long-timescale dynamics of molecular systems. An MSM is essentially a kinetic
map of the conformational space a molecule explores. The model consists of (1)
a set of conformational states and (2) a matrix of transition probabilities
(or, equivalently, transition rates) between each pair of states.

In mixtape, you can use :class:`MarkovStateModel` to build MSMs from "labeled"
trajectories -- that is, sequences of integers corresponding to the index of
the conformational state occupied by the system at each time point on a
trajectory. The :ref:`cluster` module provides a number of different methods for
clustering the trajectories that you can use to define the states.


Algorithms
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MarkovStateModel


Tradeoffs and Parameter Selection
---------------------------------
The most important tradeoff with MSMs is a `bias-variance dilemma
<http://en.wikipedia.org/wiki/Bias%E2%80%93variance_dilemma>`_ on the number of
states. We know analytically that the expected value of the relaxation
timescales is below the true value when using a finite number of states, and
that the magnitude of this bias decreases as the number of states goes up. On
the other hand, the statistical error in the MSM (variance) goes up as the
number of states increases with a fixed data set, because there are fewer
transitions (data) per element of the transition probability matrix.

There are no existing algorithms in the MSM literature which fully balance these
competing sources of error in an automatic and practical way, although some
partially satisfactory algorithms are available. [#f3]_ [#f4]_

A second key parameter is the lag time of the model. The lag time controls a
trade off between accuracy and descriptive power. [TODO: WRITE MORE]


Analyzing an MSM
----------------
1. timescales
2. eigenvectors
3. lumping


References
----------
.. [#f1] Prinz, J.-H., et al. `Markov models of molecular kinetics: Generation and validation <http://dx.doi.org/10.1063/1.3565032>`_ J. Chem. Phys. 134.17 (2011): 174105.
.. [#f2] Pande, V. S., K. A. Beauchamp, and G. R. Bowman. `Everything you wanted to know about Markov State Models but were afraid to ask <http://dx.doi.org/10.1016/j.ymeth.2010.06.002>`_ Methods 52.1 (2010): 99-105.
.. [#f3] McGibbon, R. T., C. R. Schwantes, and Vijay S. Pande. `Statistical Model Selection for Markov Models of Biomolecular Dynamics. <http://dx.doi.org/10.1021/jp411822r> J. Phys. Chem. B (2014).
.. [#f4] Kellogg, E. H., O. F. Lange, and D. Baker. `Evaluation and optimization of discrete state models of protein folding. <http:/dx.doi.org//10.1021/jp3044303>`_ J. Phys. Chem. B 116.37 (2012): 11405-11413.



