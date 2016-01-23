#!/bin/env python
from __future__ import absolute_import, division
from multiprocessing import Pool, cpu_count
from msmbuilder.utils import list_of_1d
from sklearn.utils import resample
from ..core import _MappingTransformMixin
from sklearn.base import clone
from ..msm import MarkovStateModel
import numpy as np

class BootStrapMarkovStateModel(_MappingTransformMixin):
    """Bootstrap MSM class which estimates a distribution
    of transition matrices using random sampling with replacement
    over the set of input trajectories.

    Parameters
    ----------
    n_samples : number of samples to obtain
    n_procs : number of processes to use=

    Attributes
    ----------
    all_populations_ : list of all population vectors
    mapped_populations_: array (n_samples, mle.n_states_)
    mapped_populations_mean_:array of mean of population
    mapped_populations_std_:
    mapped_populations_sem_:
     Notes
    -----
    BootStrapMarkovStateModel is a subclass of MarkovStateModel.

    """
    def __init__(self, n_samples=10,  n_procs=None, **kwargs):
        self.n_samples = n_samples
        self.n_procs = n_procs
        self.mle = MarkovStateModel(**kwargs)

        self.all_populations_ = None
        self.mapped_populations_ = None

    def _fit_one(self, sequences):
        #not sure if i need to actually clone the original mdl but it seems
        #like a safe bet
        mdl = clone(self.mle)
        mdl.fit(sequences)
        # solve the eigensystem
        mdl.eigenvalues_[0]
        return mdl

    @staticmethod
    def _mapped_populations(mdl1, mdl2):
        """
        Method to get the populations for states in mdl 1
        from populations inferred in mdl 2. Resorts to 0
        if population is not present.
        """
        return_vect = np.zeros(mdl1.n_states_)
        for i in range(mdl1.n_states_):
            try:
                #there has to be a better way to do this
                mdl1_unmapped = mdl1.inverse_transform([i])[0][0]
                mdl2_mapped = mdl2.mapping_[mdl1_unmapped]
                return_vect[i] =  mdl2.populations_[mdl2_mapped]
            except:
                pass
        return return_vect

    def _parallel_fit(self, sequences):
        """
        :param sequences:
        :return:
        """
        if self.n_procs is None:
            self.n_procs = cpu_count()
        pool = Pool(self.n_procs)


        self.all_populations_ = []
        self.mapped_populations_ = np.zeros((self.n_samples, self.mle.n_states_))

        jbs = [resample(sequences) for i in range(self.n_samples)]
        all_mdls = pool.map(self._fit_one, jbs)

        for mdl_indx, mdl in enumerate(all_mdls):
            self.all_populations_.append(mdl.populations_)
            self.mapped_populations_[mdl_indx,:] = self._mapped_populations(self.mle, mdl)

    @property
    def mapped_populations_mean_(self):
        return np.mean(self.mapped_populations_, axis=0)

    @property
    def mapped_populations_std_(self):
        return np.std(self.mapped_populations_, axis=0)


    @property
    def mapped_populations_sem_(self):
        return np.std(self.mapped_populations_, axis=0)/self.n_samples


    def fit(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        self.mle.fit(sequences, y=y)
        self._parallel_fit(sequences)

    def transform(self):
        raise NotImplementedError("This has not been implemented")

