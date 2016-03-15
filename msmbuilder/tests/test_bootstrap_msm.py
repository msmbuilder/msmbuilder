from __future__ import print_function, division
from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm.validation import BootStrapMarkovStateModel
from msmbuilder.msm.validation.bootstrapmsm import _mapped_populations as mapper
from mdtraj.testing import eq
import numpy as np



def test_mdl():
    mdl = BootStrapMarkovStateModel(n_samples=10, n_procs=2, msm_args={'lag_time': 10})


def test_resampler():
    sequences = [np.random.randint(20, size=100) for _ in range(100)]
    mdl = BootStrapMarkovStateModel(n_samples=5, n_procs=2, msm_args={'lag_time': 10})
    #probability that
    mdl.fit(sequences)
    #given a size of 100 input trajectories the probability that
    # we re-pick the original set is about (1/100)^100.
    # we test that the set of unique traj ids is never equal to
    #original 100 sets in all 5 samples
    for i in mdl.resample_ind_:
        assert len(np.unique(i)) != 100


def test_mle_eq():
    seq = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]]
    mle_mdl = MarkovStateModel(lag_time=1)
    b_mdl = BootStrapMarkovStateModel(n_samples=10, n_procs=2, msm_args={'lag_time': 1})
    mle_mdl.fit(seq)
    b_mdl.fit(seq)
    #make sure we have good model
    eq(mle_mdl.populations_, b_mdl.mle_.populations_)
    eq(mle_mdl.timescales_, b_mdl.mle_.timescales_)


def test_score():
    seq = [np.random.randint(20, size=100),
            np.random.randint(20, size=100),
            np.random.randint(20, size=100)]
    bmsm = BootStrapMarkovStateModel(n_samples=10, n_procs=2, msm_args={'lag_time':1})
    bmsm.fit(seq)
    # test that all samples got a training score ...
    assert np.array(bmsm.all_training_scores_).shape[0] == 10
    # ... and that the training score wasn't NaN
    assert sum(np.isnan(bmsm.all_training_scores_)) == 0
    # test that a test score was attempted (OK if it's NaN)
    assert bmsm.n_samples == np.array(bmsm.all_test_scores_).shape[0]


class fakemsm(object):
    def __init__(self, n_states, mapping, populations ):
        self.n_states_ = n_states
        self.mapping_ = mapping
        self.pop = populations


    @property
    def populations_(self):
        return [self.pop[i] for i in self.mapping_.keys()]

    def inverse_transform(self,ind):
        for k in self.mapping_.keys():
            if self.mapping_[k]==ind[0]:
                return [[k]]
        return None


def test_mapper_1():
    #base case
    mdl1_mapping = {0:0, 1:1, 2:2}
    mdl2_mapping = {0:0, 1:1, 2:2}

    pop = {0:0.1, 1:0.2, 2:0.3}
    mdl1 = fakemsm(3, mdl1_mapping,pop)
    mdl2 = fakemsm(3, mdl2_mapping,pop)

    mapped_pop = mapper(mdl1, mdl2)

    assert(mapped_pop== [0.1, 0.2, 0.3]).all()

def test_mapper_2():
    #case where state 2 is throw out
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:0,1:1}

    pop = {0:0.1, 1:0.2, 2:0.3}

    mdl1 = fakemsm(3, mdl1_mapping,pop)
    mdl2 = fakemsm(2, mdl2_mapping,pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.1, 0.2, 0.0]).all()

def test_mapper_3():
    #case where state 1 is throw out
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:0,2:1}

    pop = {0:0.1, 1:0.2, 2:0.3}
    mdl1 = fakemsm(3, mdl1_mapping, pop)
    mdl2 = fakemsm(2, mdl2_mapping, pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.1, 0, 0.3]).all()

def test_mapper_4():
    #case where the mdl is jumbled around
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:1,1:2,2:0}

    pop = {0:0.1, 1:0.2, 2:0.3}
    mdl1 = fakemsm(3, mdl1_mapping, pop)
    mdl2 = fakemsm(2, mdl2_mapping, pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.2, 0.3, 0.1]).all()
