#!/bin/env python
# from __future__ import absolute_import, division
from multiprocessing import Pool, cpu_count
import numpy as np
from msmbuilder.utils import list_of_1d
from msmbuilder.base import BaseEstimator
import copy
from sklearn.utils import resample
import itertools

class BootstrapModel(BaseEstimator):
    """Bootstrap Model class which estimates a distribution
    of transition matrices using random sampling with replacement
    over the set of input trajectories.

    Parameters
    ----------
    estimator : msmbuilder object
        MSM or other estimator
    n_samples : number of samples to obtain
    n_procs : number of processes to use=

    Attributes
    ----------
    all_n_states_ : list
        List of the number of states in each of the boot strapped model
    all_mapping_ : list of dict
        List of mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessarily be integers in (0, ..., n_states_ - 1), for
        example. The semantics of ``mapping_[i] = j`` is that state ``i`` from
        the "input space" is represented by the index ``j`` in this MSM.
    all_countsmat_ : list of arrays where each array has the
        shape = (n_states_, n_states_). The indices `i` and `j` are
        the "internal" indices described above.
    all_transmats_ : list of arrays where each array has the
        shape = (n_states_, n_states_). The indices `i` and `j` are
        the "internal" indices described above.

    """
    def __init__(self, estimator, n_samples, n_procs=None):
        self.estimator = estimator
        self.n_samples = n_samples
        self.n_procs = n_procs

        self.all_mdls = None
        self.all_mapping_ = None
        self.all_populations_ = None
        self.all_countsmat_ = None
        self.all_transmats_ = None
        self.all_n_states_ = None

    def _fit_one(self, sequences):
        mdl = copy.deepcopy(self.estimator)
        mdl.fit(sequences)
        # solve the eigensystem
        mdl.eigenvalues_[0]
        return mdl

    def _parallel_fit(self, sequences):
        """
        :param sequences:
        :return:
        """
        if self.n_procs is None:
            self.n_procs = cpu_count()
        pool = Pool(self.n_procs)

        all_mdls = pool.map(self._fit_one,
                            itertools.repeat(resample(sequences), self.n_samples))

        self.all_mdls = all_mdls
        self.all_mapping_ =  []
        self.all_populations_ = []
        self.all_countsmat_ = []
        self.all_transmats_ = []
        self.all_n_states_ = []

        for mdl in all_mdls:
            self.all_n_states_.append(mdl.n_states_)
            self.all_mapping_.append(mdl.mapping_)
            self.all_populations_.append(mdl.populations_)
            self.all_countsmat_.append(mdl.countsmat_)
            self.all_transmats_.append(mdl.transmat_)

    def fit(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        self._parallel_fit(sequences)

    def transform(self):
        raise NotImplementedError("This has not been implemented")

