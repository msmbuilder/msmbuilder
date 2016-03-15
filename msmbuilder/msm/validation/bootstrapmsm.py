# Author: Mohammad M. Sultan <msultan@stanford.edu>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.

from __future__ import absolute_import, division
from multiprocessing import Pool, cpu_count
from msmbuilder.utils import list_of_1d
from sklearn.utils import resample
from ..core import _MappingTransformMixin
from ..msm import MarkovStateModel
import numpy as np
import warnings

class BootStrapMarkovStateModel(_MappingTransformMixin):
    """Bootstrap MarkovState Model.

    This model fits a series of first-order Markov models
    to bootstrap samples obtained from a dataset of
    integer-valued timeseries.The sequence of transition
    matrices are obtained using random sampling with
    replacement over the set of input trajectories.
    The model also fits the mle over the original set.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap models to construct
    n_procs : int
        Number of processors to use.
        Defaults to int(cpu_count/2)
    msm_args: dict
        Dictionary containing arguments to pass unto
        the MSM models.

    Attributes
    ----------
    mle_ : Markov State Model
        MSM model fit unto original dataset. The state mapping
        inferred here is used through the rest of the models.
    all_populations_ : list of lists
        Array of all populations obtained from each model
    all_training_scores_ : list
        List of scores obtained from each model
    all_test_scores_ : list
        List of scores obtained on the sequences omitted from
        each model
    resample_ind_ : list
        lis of resample indices used to fit each bootstrap mdl.
        This can be used to regenerate any mdl without having
        to store it inside this object.
    mapped_populations_ : array (n_samples, mle.n_states_)
        Array containing population estimates from all the
        models for the states retained in the mle model.
    mapped_populations_mean_ : array shape = (mle.n_states_)
        Mean population across the set of models for the states
        contained in the mle model.
    mapped_populations_std_ : array shape = (mle.n_states_)
        Population std across the set of models for the states
        contained in the mle model.
    mapped_populations_sem_ : array shape = (mle.n_states_)
        Population sem across the set of models for the states
        contained in the mle model.(std/sqrt(self._succesfully_fit))
    training_scores_mean_ :  list
        Mean population across the list of model scores
    training_scores_std_ : list
        Population std across the list of model scores
    test_scores_mean_ : list
        Mean population across the list of scores obtained on
        the sequences omitted from each model
    test_scores_std : list
        Population std across the list of scores obtained on
        the sequences omitted from each model 

    Notes
    -----
    The correct number of bootstrap sample is subject to
    debate with several hundred samples(n_samples)
    being the recommended starting figure.

    The fit function for this model optionally takes in a pool
    of workers making it capable of parallelizing across
    compute nodes via mpi or ipyparallel. This can lead to
    a significant speed up for larger number of samples.

    Examples
    --------
    >>> bmsm = BootStrapMarkovStateModel(n_samples=800,
                    msm_args={'lag_time':1})
    """
    def __init__(self, n_samples=10,  n_procs=None, msm_args={}):
        self.n_samples = n_samples
        self.n_procs = n_procs
        self.msm_args = msm_args
        self.mle_ = MarkovStateModel(**self.msm_args)

        self._succesfully_fit = 0
        self._ommitted_trajs_ = None
        self.all_populations_ = None
        self.mapped_populations_ = None
        self.all_training_scores_ = None
        self.all_test_scores_ = None
        self.resample_ind_ = None


    def fit(self, sequences, y=None, pool=None):
        sequences = list_of_1d(sequences)
        self.mle_.fit(sequences, y=y)
        self._parallel_fit(sequences, pool)


    def _parallel_fit(self, sequences, pool=None):

        if self.n_procs is None:
            self.n_procs = int(cpu_count()/2)
        if pool is None:
            pool = Pool(self.n_procs)

        self.all_populations_ = []
        self.mapped_populations_ = np.zeros((self.n_samples, self.mle_.n_states_))
        self.all_training_scores_ = []
        self.all_test_scores_ = []

        #we cache the sequencs of re sampling indices so that any mdl can be
        #regenerated later on
        self.resample_ind_ = [resample(range(len(sequences)))
                                 for _ in range(self.n_samples)]

        jbs =[([sequences[trj_ind] for trj_ind in sample_ind],
               self.msm_args)
               for sample_ind in self.resample_ind_]

        traj_set = set(range(len(sequences)))

        #get trajectory index that were omitted in each sampling 
        omitted_trajs = [traj_set.difference(set(sample_ind))
                            for sample_ind in self.resample_ind_]

        self._ommitted_trajs_ = omitted_trajs

        #get the test jobs
        test_jbs = [[sequences[trj_ind] for trj_ind in omitted_index]
                    for omitted_index in omitted_trajs]

        all_mdls = pool.map(_fit_one, jbs)

        for mdl_indx, mdl in enumerate(all_mdls):
            if mdl is not None:
                self._succesfully_fit += 1
                self.all_populations_.append(mdl.populations_)
                self.mapped_populations_[mdl_indx,:] = \
                    _mapped_populations(self.mle_, mdl)
                self.all_training_scores_.append(mdl.score_) # BEH
                try:
                    self.all_test_scores_.append(mdl.score(test_jbs[mdl_indx]))
                except ValueError:
                    self.all_test_scores_.append(np.nan)

        return


    @property
    def mapped_populations_mean_(self):
        return np.mean(self.mapped_populations_, axis=0)

    @property
    def mapped_populations_std_(self):
        return np.std(self.mapped_populations_, axis=0)

    @property
    def mapped_populations_sem_(self):
        return np.std(self.mapped_populations_, axis=0)/np.sqrt(self._succesfully_fit)

    @property
    def training_scores_mean_(self):
        return np.nanmean(self.all_training_scores_, axis=0)

    @property
    def training_scores_std_(self):
        return np.nanstd(self.all_training_scores_, axis=0)

    @property
    def test_scores_mean_(self):
        return np.nanmean(self.all_test_scores_, axis=0)

    @property
    def test_scores_std_(self):
        return np.nanstd(self.all_test_scores_, axis=0)


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



def _fit_one(jt):
    sequences, msm_args = jt
    mdl = MarkovStateModel(**msm_args)
    #there is no guarantee that the mdl fits this sequence set so
    #we return None in that instance.
    try:
        mdl.fit(sequences)
        # solve the eigensystem
    except ValueError:
        mdl = None
        warnings.warn("One of the MSMs fitting "
                          "failed")
    return mdl