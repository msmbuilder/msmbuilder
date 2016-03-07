MSMBuilder
==========

[![Build Status](https://travis-ci.org/msmbuilder/msmbuilder.svg?branch=master)] (https://travis-ci.org/msmbuilder/msmbuilder)
[![PyPi version](https://badge.fury.io/py/msmbuilder.svg)]                       (https://pypi.python.org/pypi/msmbuilder/)
[![License](https://img.shields.io/badge/license-LGPLv2.1+-red.svg?style=flat)]  (https://pypi.python.org/pypi/msmbuilder/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)] (http://msmbuilder.org)

MSMBuilder is a python package which implements a series of statistical
models for high-dimensional time-series. It is particularly focused on the
analysis of atomistic simulations of biomolecular dynamics. For example,
MSMBuilder has been used to model protein folding and conformational change
from molecular dynamics (MD) simulations. MSMBuilder is available under the
LGPL (v2.1 or later).

Capabilities include:

- Feature extraction into dihedrals, contact maps, and more
- Geometric clustering with a variety of algorithms.
- Dimensionality reduction using time-structure independent component
  analysis (tICA) and principal component analysis (PCA).
- Markov state model (MSM) construction
- Rate-matrix MSM construction
- Hidden markov model (HMM) construction
- Timescale and transition path analysis.

Check out the documentation at [msmbuilder.org](http://msmbuilder.org) and
join the [mailing list](https://mailman.stanford.edu/mailman/listinfo/msmbuilder-user)

Installation
------------

The preferred installation mechanism for `msmbuilder` is with `conda`:

```bash
$ conda install -c omnia msmbuilder
```

If you don't have conda, or are new to scientific python, we recommend that
you download the [Anaconda scientific python distribution](https://store.continuum.io/cshop/anaconda/).


Workflow
--------

An example workflow might be as follows:

1. Set up a system for molecular dynamics, and run one or more simulations
   for as long as you can on as many CPUs or GPUs as you have access to.
   There are a lot of great software packages for running MD, e.g [OpenMM]
   (https://simtk.org/home/openmm), [Gromacs](http://www.gromacs.org/),
   [Amber](http://ambermd.org/), [CHARMM](http://www.charmm.org/), and
   many others. MSMBuilder is not one of them.

2. Transform your MD coordinates into an appropriate set of features.

3. Perform some sort of dimensionality reduction with tICA or PCA.
   Reduce your data into discrete states by using clustering.

4. Fit an MSM, rate matrix MSM, or HMM. Perform model selection using
   cross-validation with the [generalized matrix Rayleigh quotient](http://arxiv.org/abs/1407.8083)
