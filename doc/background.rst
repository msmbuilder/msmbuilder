.. _background:

Background and Motivation
-------------------------

The goals
~~~~~~~~~

The aim of this package is to provide software tools for predictive modeling of the **long timescale dynamics** of biomolecular systems using **statistical modeling to analyze physical simulations**.

The basic question we're interested in is, "given a dataset of one or more stochastic trajectories tracking the coordinates of every (>10,000+) atom in a molecular system at a discrete time interval, how do we understand the slow dynamical processes and make quantitative predictions about the system?"


Workflow
~~~~~~~~

1. Set up a system for molecular dynamics, and run one or more simulations for as long as you can on as many CPUs or GPUs as you have access to.
  - There are a lot of great software packages for running MD, e.g `OpenMM <https://simtk.org/home/openmm>`_, `Gromacs <http://www.gromacs.org/>`_, `Amber <http://ambermd.org/>`_, `CHARMM <http://www.charmm.org/>`_, and many others. Mixtape is not one of them.
2. Analyze your simulations with Mixtape.
3. Profit.

