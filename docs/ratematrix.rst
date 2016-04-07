.. _ratematrix:
.. currentmodule:: msmbuilder.msm

Continuous-time MSMs
====================

:class:`MarkovStateModel` estimates a series of
transition *probabilities* among states that depend on the discrete
lag-time. Physically, we are probably more interested in a sparse set of
transition *rates* in and out of states, estimated by
:class:`ContinuousTimeMSM`.


Theory
------

Consider an `n`-state time-homogeneous Markov process, :math:`X(t)`. At
time :math:`t`, the :math:`n`-vector :math:`P(t) = Pr[ X(t) = i ]` is the
probability that the system is in each of the :math:`n` states. These
probabilities evolve forward in time, governed by an :math:`n \times n`
transition rate matrix :math:`K`

.. math ::
    dP(t)/dt = P(t) \cdot K

The solution is

.. math ::
    P(t) = \exp(tK) \cdot P(0)

Where :math:`\exp(tK)` is the matrix exponential. Written differently, the
state-to-state lag-:math:`\tau` transition probabilities are

.. math ::
    Pr[ X(t+\tau) = j \;|\; X(t) = i ] = \exp(\tau K)_{ij}

For this model, we observe the evolution of one or more chains,
:math:`X(t)` at a regular interval, :math:`\tau`. Let :math:`C_{ij}` be the
number of times the chain was observed at state :math:`i` at time :math:`t`
and at state :math:`j` at time :math:`t+\tau` (the number of observed
transition counts). Suppose that :math:`K` depends on a parameter vector,
:math:`\theta`. The log-likelihood is

.. math ::
  \mathcal{L}(\theta) = \sum_{ij} \left[
      C_{ij} \log\left(\left[\exp(\tau K(\theta))\right]_{ij}\right)\right]

The :class:`ContinuousTimeMSM` model finds a rate matrix that fits the data
by maximizing this likelihood expression.  Specifically, it uses L-BFGS-B
to find a maximum likelihood estimate (MLE) rate matrix,
:math:`\hat{\theta}` and :math:`K(\hat{\theta})`.

Uncertainties
~~~~~~~~~~~~~

Analytical estimates of the asymptotic standard deviation in estimated
parameters like the stationary distribution, rate matrix, eigenvalues, and
relaxation timescales can be computed by calling methods on the
:class:`ContinuousTimeMSM` object. See [1] for more detail.


Algorithms
----------

.. autosummary::
    :toctree: _ratematrix/

    ContinuousTimeMSM


References
----------
.. [1] McGibbon, R. T. and V. S. Pande, "Efficient maximum likelihood parameterization
   of continuous-time Markov processes." J. Chem. Phys. 143 034109 (2015) http://dx.doi.org/10.1063/1.4926516
.. [2] Kalbfleisch, J. D., and Jerald F. Lawless. "The analysis of panel data
   under a Markov assumption." J. Am. Stat. Assoc. 80.392 (1985): 863-871.

.. vim: tw=75
