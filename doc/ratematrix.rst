.. currentmodule:: msmbuilder.msm

Continuous-time Markov State Model
==================================

Theory
------

Consider an `n`-state time-homogeneous Markov process, :math:`X(t)` At time
:math:`t`, the :math:`n`-vector :math:`P(t) = Pr[ X(t) = i ]` is probability that the system is in each of the :math:`n` states. These probabilities
evolve forward in time, governed by an :math:`n \times n` transition rate matrix :math:`K` 

.. math ::
    dP(t)/dt = P(t) \cdot K

The solution is

.. math ::
    P(t) = \exp(tK) \cdot P(0)

Where :math:`\exp(tK)` is the matrix exponential. Written differently, the state-to-state lag-:math:`\tau` transition probabilities are

.. math ::
    Pr[ X(t+\tau) = j \;|\; X(t) = i ] = exp(\tau K)_{ij}

For this model, we observe the evolution of one or more chains, :math:`X(t)` at
a regular interval, :math:`\tau`. Let :math:`C_{ij}` be the number of times the
chain was observed at state :math:`i` at time :math:`t` and at state :math:`j` at time :math:`t+\tau` (the number of observed transition counts). Suppose that
:math:`K` depends on a length-:math:`b` parameter vector, :math:`\theta`. The log-likelihood is

.. math ::
  \mathcal{L}(\theta) = \sum_{ij} \left[
      C_{ij} \log\left(\left[\exp(\tau K(\theta))\right]_{ij}\right)\right]

Dense Parameterization
----------------------
This code parameterizes :math:`K(\theta)` such that :math:`K` is constrained
to satisfy `detailed balance <http://en.wikipedia.org/wiki/Detailed_balance>`_
with respect to a stationary distribution, :math:`pi`. For an :math:`n`-state
model, :math:`\theta` is of length :math:`n(n-1)/2 + n`. The first
:math:`n(n-1)/2` elements of :math:`theta` are the parameterize the symmetric
rate matrix, :math:`S`, and the remaining :math:`n` entries parameterize the
stationary distribution. For :math:`n=4`, the function :math:`K(\theta)` is

.. math ::
    \begin{array}a
    s_u = \exp(\theta_u)  &\text{for} &u = 0, 1, \ldots, n(n-1)/2 \\
    \pi_i = \exp(\theta_{i + n(n-1)/2}) &\text{for} &i = 0, 1, ..., n \\
    r_i = \sqrt{\pi_i}     &\text{for} &i = 0, 1, ..., n
    \end{array}

.. math ::
    K(\theta) = \begin{bmatrix}
    k_{00}        &  s_0(r_1/r_0)   &   s_1(r_2/r_0)  &  s_2(r_3/r_0)  \\
    s_0(r_0/r_1)  &    k_{11}       &   s_3(r_2/r_1)  &  s_4(r_3/r_1) \\
    s_1(r_0/r_2)  &  s_3(r_1/r_2)   &     k_{22}      &  s_5(r_3/r_2) \\
    s_2(r_0/r_3)  &  s_4(r_1/r_3)   &   s_5(r_2/r_3)  &    k_{33}
    \end{bmatrix}

where the diagonal elements :math:`k_{ii}` are set such that the row sums of
:math:`K` are all zero, :math:`k_{ii} = -\sum_{j != i} K_{ij}`. This form for :math:`K` satisfies detailed balance by construction

.. math ::
  K_{ij} / K_{ji} =  r_j^2 / r_i^2  =  \pi_j / \pi_i

.. math ::
  \pi_i K_{ij} = pi_j K_{ji}

Note that :math:`K` is built from :math:`\exp(\theta)`. This parameterization
makes optimization easier, since it keeps the off-diagonal entries positive
and the diagonal entries negative. The optimization with respect to :math:`\theta` without constraints.

Sparse Parameterization
-----------------------
Using the dense parameterization, it is not possible for elements of :math:`K`
to be exactly zero, because the symmetric rate matrix is parameterized in
log-space. Thus, :class:`ContinuousTimeMSM` also includes an ability to find a
sparse rate matrix, through an alternative parameterization. In the sparse
parameterization, certain entries in :math:`\theta` are fixed at :math:`-\infty`,
such that :math:`s_u = 0`. The indices of the "active" elements in
:math:`\theta` are stored in an array of indices, ``inds``, and a compressed
representation of `\theta` is used, with only the "active" elements explicitly.

Estimation
----------
:class:`ContinuousTimeMSM` uses L-BFGS-B to find a maximum likelihood estimate
(MLE) rate matrix, :math:`\hat{\theta}` and :math:`K(\hat{\theta})`. By default,
the algorithm works first be estimating a fully dense rate matrix. Then, small
off-diagonal elements of K are taken as candidates for truncation to zero. A
new optimization using the sparse parameterization is performed with these elements constrained. If the log-likelihood of the sparse model is superior to
the log likelihood of the dense model, it is retained.

Algorithms
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ContinuousTimeMSM

References
----------
..[1] Kalbfleisch, J. D., and Jerald F. Lawless. "The analysis of panel data
under a Markov assumption." J. Am. Stat. Assoc. 80.392 (1985): 863-871.
