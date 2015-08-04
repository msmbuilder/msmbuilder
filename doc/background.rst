.. _background:

Background and Motivation
-------------------------

The aim of this package is to provide software tools for predictive
modeling of the long timescale dynamics of biomolecular systems using
statistical modeling to analyze physical simulations.

Given a dataset of one or more stochastic trajectories tracking the
coordinates of every (>10,000+) atom in a molecular system at a discrete
time interval, how do we understand the slow dynamical processes and make
quantitative predictions about the system?


Workflow
~~~~~~~~

A default workflow is outlined below. Note that most steps are optional
under certain circumstances. Read the rest of the documentation to
understand when.

1. Set up a system for molecular dynamics, and run one or more simulations
   for as long as you can on as many CPUs or GPUs as you have access to.
   There are a lot of great software packages for running MD, e.g `OpenMM
   <https://simtk.org/home/openmm>`_, `Gromacs <http://www.gromacs.org/>`_,
   `Amber <http://ambermd.org/>`_, `CHARMM <http://www.charmm.org/>`_, and
   many others. MSMBuilder is not one of them.

2. :ref:`Featurize<featurization>` trajectories into an appropriate vector of
   features such as backbone dihedral angles.

3. :ref:`Reduce the dimensionality<decomposition>` of your features using
   tICA or a similar algorithm.

4. :ref:`Cluster<cluster>` your data to define (micro-)states.

5. :ref:`Estimate a model<msm>` of your data.

6. Use :ref:`GMRQ cross-validation<gmrq>` to select the best model.

.. todo:: Please update the above link to gmrq


.. figure:: _static/flow-chart.png
    :align: center
    :width: 80%

    A diagram of potential workflows.

.. vim: tw=75
