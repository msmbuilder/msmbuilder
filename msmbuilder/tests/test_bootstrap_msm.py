from __future__ import print_function, division
import numpy as np
from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm.validation import BootStrapMarkovStateModel
from msmbuilder.msm.validation.bootstrapmsm import _mapped_populations as mapper
from mdtraj.testing import eq



def test_mdl():
    mdl = BootStrapMarkovStateModel(n_samples=10, n_procs=2, lag_time=10)
    return True


def test_mle_eq():
    seq = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]]
    mle_mdl = MarkovStateModel(lag_time=1)
    b_mdl = BootStrapMarkovStateModel(n_samples=10, n_procs=2, lag_time=1)
    mle_mdl.fit(seq)
    b_mdl.fit(seq)
    #make sure we have good model
    eq(mle_mdl.populations_, b_mdl.mle.populations_)
    eq(mle_mdl.timescales_, b_mdl.mle.timescales_)
    return True


class fakemsm():
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
    return True

def test_mapper_2():
    #case where state 2 is throw out
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:0,1:1}

    pop = {0:0.1, 1:0.2, 2:0.3}

    mdl1 = fakemsm(3, mdl1_mapping,pop)
    mdl2 = fakemsm(2, mdl2_mapping,pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.1, 0.2, 0.0]).all()
    return True

def test_mapper_3():
    #case where state 1 is throw out
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:0,2:1}

    pop = {0:0.1, 1:0.2, 2:0.3}
    mdl1 = fakemsm(3, mdl1_mapping, pop)
    mdl2 = fakemsm(2, mdl2_mapping, pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.1, 0, 0.3]).all()
    return True

def test_mapper_4():
   #case where the mdl is jumbled around
    mdl1_mapping = {0:0,1:1,2:2}
    mdl2_mapping = {0:1,1:2,2:0}

    pop = {0:0.1, 1:0.2, 2:0.3}
    mdl1 = fakemsm(3, mdl1_mapping, pop)
    mdl2 = fakemsm(2, mdl2_mapping, pop)

    mapped_pop = mapper(mdl1, mdl2)
    assert (mapped_pop==[0.2, 0.3, 0.1]).all()
    return True