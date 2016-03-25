.. _msm:
.. currentmodule:: msmbuilder.msm

Markov state models (MSMs)
==========================

Markov state models (MSMs) are a class of models for modeling the
long-timescale dynamics of molecular systems. They model the dynamics of a
system as a series of memoryless, probabilistic jumps between a set of
states. Practically, the model consists of (1) a set of conformational
states, and (2) a matrix of transition probabilities between each pair of
states.

In MSMBuilder, you can use :class:`MarkovStateModel` to build MSMs from
"labeled" trajectories -- that is, sequences of integers that are the
result of :ref:`clustering<cluster>`.

Algorithms
----------

.. autosummary::
    :toctree: _msm/

    MarkovStateModel
    BayesianMarkovStateModel

Maximum Likelihood and Bayesian Estimation
------------------------------------------

There are two steps in constructing an MSM

#. Count the number of observed transitions between states. That is,
   construct :math:`\mathbf{C}` such that :math:`C_{ij}` is the number of
   observed transitions from state :math:`i` at time :math:`t` to state
   :math:`j` at time :math:`t+\tau`, summed over all times :math:`t`.

#. Estimate the transition probability matrix, :math:`\mathbf{T}`

   .. math ::
       T_{ij} = P( s_{t+\tau} = j | s_t = i)
    
   where :math:`S = (s_t)` is a trajectory in state-index space of length
   :math:`N`, and :math:`s_t \in \{1, \ldots, k\}` the state-index of the
   trajectory at time :math:`t`.

The probability that a given transition probability matrix would generate
some observed trajectory (the likelihood) is

.. math ::
  \mathcal{L}(\mathbf{T}) = P(S | \mathbf{T}) =
  \prod_{t=0}^{N-\tau} T_{s_t, s_{t+\tau}} = \prod_{i,j}^{k} T_{ij}^{C_{ij}}.

Assuming a prior distribution on :math:`T` of the form
:math:`P(T)=\prod_{ij} T_{ij}^{B_{ij}}`, we then have a posterior
distribution

.. math ::
   P(\mathbf{T} | S) \propto \prod_{i,j}^{k} T_{ij}^{B_{ij} + C_{ij}}.


MSMBuilder implements two MSM estimators.

- :class:`MarkovStateModel` performs maximum likelihood estimation.  It
  estimates a single transition matrix, :math:`\mathbf{T}`, to maximimize
  :math:`\mathcal{L}(\mathbf{T})`.

- :class:`BayesianMarkovStateModel` uses Metropolis Markov chain Monte
  Carlo to (approximately) draw a sample of transition matrices from the
  posterior distribution :math:`P(\mathbf{T} | S)`. This sampler is
  described in Metzner et al. [#f5]_ This can be used to estimate the
  sampling uncertainty in functions of the transition matrix (e.g.
  relaxation timescales).

.. note::

   The uncertainty in the transition matrix (and functions of the
   transition matrix) that can be estimated from
   :class:`BayesianMarkovStateModel` do not **fully** account for all
   sources of error. In particular, the discretization induced by
   clustering produces a negative bias on  the eigenvalues of the
   transition matrix -- they asymptotically underestimate the eigenvalues
   of the propagator / transfer operator in the limit of infinite sampling.
   [#f6]_ See section 3D (Quantifying the discretization error) of Prinz et
   al. for more discussion on the discretization error. [#f1]_


Tradeoffs and Parameter Selection
---------------------------------

The most important tradeoff with MSMs is a `bias-variance dilemma
<http://en.wikipedia.org/wiki/Bias%E2%80%93variance_dilemma>`_ on the
number of states. We know analytically that the expected value of the
relaxation timescales is below the true value when using a finite number of
states, and that the magnitude of this bias decreases as the number of
states goes up. On the other hand, the statistical error in the MSM
(variance) goes up as the number of states increases with a fixed data set,
because there are fewer transitions (data) per element of the transition
probability matrix.

There are no existing algorithms in the MSM literature which fully balance
these competing sources of error in an automatic and practical way,
although some partially satisfactory algorithms are available. [#f3]_
[#f4]_

.. todo: talk about lag time

.. todo: analysis page


References
----------
.. [#f1] Prinz, J.-H., et al. `Markov models of molecular kinetics: Generation and validation <http://dx.doi.org/10.1063/1.3565032>`_ J. Chem. Phys. 134.17 (2011): 174105.
.. [#f2] Pande, V. S., K. A. Beauchamp, and G. R. Bowman. `Everything you wanted to know about Markov State Models but were afraid to ask <http://dx.doi.org/10.1016/j.ymeth.2010.06.002>`_ Methods 52.1 (2010): 99-105.
.. [#f3] McGibbon, R. T., C. R. Schwantes, and Vijay S. Pande. `Statistical Model Selection for Markov Models of Biomolecular Dynamics. <http://dx.doi.org/10.1021/jp411822r>`_ J. Phys. Chem. B (2014).
.. [#f4] Kellogg, E. H., O. F. Lange, and D. Baker. `Evaluation and optimization of discrete state models of protein folding. <http://dx.doi.org//10.1021/jp3044303>`_ J. Phys. Chem. B 116.37 (2012): 11405-11413.
.. [#f5] Metzner, P., F. Noe, and C. Schutte. `Estimating the sampling error: Distribution of transition matrices and functions of transition matrices for given trajectory data. <http://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.021106>`_ Phys. Rev. E 80.2 (2009): 021106.
.. [#f6] Nuske, F., et al. `Variational approach to molecular kinetics. <http://pubs.acs.org/doi/abs/10.1021/ct4009156>`_ J. Chem. Theory Comput.10.4 (2014): 1739-1752.

.. vim: tw=75
