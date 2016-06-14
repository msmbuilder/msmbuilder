.. _gmrq:
.. currentmodule:: msmbuilder

Model Selection using GMRQ
==========================

The generalized matrix Rayleigh quotient (GMRQ) is a specific application of
the variational principle (adapted from `quantum mechanics
<https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)>`_)
for Markov state models and a useful tool for model parameter selection.

The variational principle yields a rigorous way of comparing two different
Markov models for the same underlying stochastic process when using different
state decompositions. Even under the assumption that you have access to
infinite sampling, there is still some error associated with approximating the
true continuous eigenfunctions of your modeled process with the indicator
functions, as is the case with Markov state models. If we interpret the
variational theorem as the measure of the quality of this approximation, the
state decomposition that leads to a Markov model with larger leading dynamical
eigenvalues is consequently the better state decomposition. If you wish to see
the full derivation of this quantity, please refer to [#f1]_.

Using this method, we can generate single scalar-valued scores for a proposed
model given a supplied data set. This allows for the use of separate testing
and training data sets to quantify and avoid statistical overfitting.
This method extends these tools, making it possible to score trained models on
new datasets and to perform hyperparameter selection.

Algorithms
----------

.. autosummary::
    :toctree: _gmrq/

    decomposition.tICA.score
    msm.MarkovStateModel.score
    msm.ContinuousTimeMSM.score




References
----------

.. [#f1] McGibbon, Robert T., and Vijay S. Pande. `Variational cross-validation of slow dynamical modes in molecular kinetics <http://dx.doi.org/10.1063/1.4916292>`_ J. Chem. Phys. 142, 124105 (2015).

.. vim: tw=75
