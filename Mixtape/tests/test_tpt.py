from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import os

from mixtape import tpt
from mixtape.msm import MarkovStateModel

import numpy as np
import numpy.testing as npt

from mdtraj import io

def test_paths():

    net_flux = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.3, 0.0, 0.2],
                         [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    sources = np.array([0])
    sinks = np.array([4, 5])

    ref_paths = [[0, 2, 4],
                 [0, 1, 3,  5],
                 [0, 1, 5]]

    ref_fluxes = np.array([0.5, 0.3, 0.2])
    
    res_bottle = tpt.paths(sources, sinks, net_flux, remove_path='bottleneck') 
    res_subtract = tpt.paths(sources, sinks, net_flux, remove_path='subtract')

    for paths, fluxes in [res_bottle, res_subtract]:
        npt.assert_array_almost_equal(fluxes, ref_fluxes)
        assert len(paths) == len(ref_paths)

        for i in xrange(len(paths)):
            npt.assert_array_equal(paths[i], ref_paths[i])

        
def test_committors():
    
    msm = MarkovStateModel(lag_time=1)
    assignments = np.random.randint(3, size=(10, 1000))
    msm.fit(assignments)

    tprob = msm.transmat_

    committors = tpt.committors([0], [2], msm)

    # The probability of hitting state 2 before going back to state 1
    # is a sum over possible paths that don't go back to state 0.
    # Since there are only three states the paths are all something
    # of the form 1, 1, 1, 1, ..., 1, 1, 2
    # Theoretically we need infinitely many 1->1 transitions, but 
    # that approaches zero, so the approximation below is probably
    # just fine.
    ref = np.power(tprob[1, 1], np.arange(1000)).sum() * tprob[1, 2]
    ref = np.array([0, ref, 1])

    #print(committors, ref)

    npt.assert_array_almost_equal(ref, committors)


def test_cond_committors():
    # depends on tpt.committors
    
    msm = MarkovStateModel(lag_time=1)
    assignments = np.random.randint(4, size=(10, 1000))
    msm.fit(assignments)

    tprob = msm.transmat_

    for_committors = tpt.committors(0, 3, msm)
    cond_committors = tpt.conditional_committors(0, 3, 2, msm)

    # The committor for state one can be decomposed into paths that
    # do and do not visit state 2 along the way. The paths that do not
    # visit state 1 must look like 1, 1, 1, ..., 1, 1, 3. So we can
    # compute them with a similar approximation as the forward committor
    # Since we want the other component of the forward committor, we
    # subtract that probability from the forward committor
    ref = for_committors[1] - np.power(tprob[1, 1], np.arange(5000)).sum() * tprob[1, 3]
    #print (ref / for_committors[1])
    ref = [0, ref, for_committors[2], 0]

    #print(cond_committors, ref)

    npt.assert_array_almost_equal(ref, cond_committors)


def test_fluxes():
    # depends on tpt.committors

    msm = MarkovStateModel(lag_time=1)
    assignments = np.random.randint(3, size=(10, 1000))
    msm.fit(assignments)


    tprob = msm.transmat_
    pop = msm.populations_
    # forward committors
    qplus = tpt.committors(0, 2, msm)
    
    ref_fluxes = np.zeros((3, 3))
    ref_net_fluxes = np.zeros((3, 3))
    for i in xrange(3):
        for j in xrange(3):
            if i != j:
                # Eq. 2.24 in Metzner et al. Transition Path Theory. 
                # Multiscale Model. Simul. 2009, 7, 1192-1219.
                ref_fluxes[i, j] = pop[i] * tprob[i, j] * (1 - qplus[i]) * qplus[j]

    for i in xrange(3):
        for j in xrange(3):
            ref_net_fluxes[i, j] = np.max([0, ref_fluxes[i, j] - ref_fluxes[j, i]])

    fluxes = tpt.fluxes(0, 2, msm)
    net_fluxes = tpt.net_fluxes(0, 2, msm)

    #print(fluxes)
    #print(ref_fluxes)

    npt.assert_array_almost_equal(ref_fluxes, fluxes)
    npt.assert_array_almost_equal(ref_net_fluxes, net_fluxes)

def test_hubscore():
    #Make an actual hub!

    tprob = np.array([[0.8, 0.0, 0.2, 0.0, 0.0],
                      [0.0, 0.8, 0.2, 0.0, 0.0],
                      [0.1, 0.1, 0.6, 0.1, 0.1],
                      [0.0, 0.0, 0.2, 0.8, 0.0],
                      [0.0, 0.0, 0.2, 0.0, 0.8]])

    msm = MarkovStateModel(lag_time=1)
    msm.transmat_ = tprob
    msm.n_states_ = 5

    score = tpt.hub_scores(msm, 2)[0]

    assert score == 1.0


def test_harder_hubscore():
    # depends on tpt.committors and tpt.conditional_committors

    assignments = np.random.randint(10, size=(10, 1000))
    msm = MarkovStateModel(lag_time=1)
    msm.fit(assignments)
    
    hub_scores = tpt.hub_scores(msm)

    ref_hub_scores = np.zeros(10)
    for A in xrange(10):
        for B in xrange(10):
            committors = tpt.committors(A, B, msm)
            denom = msm.transmat_[A, :].dot(committors) #+ msm.transmat_[A, B]
            for C in xrange(10):
                if A == B or A == C or B == C:
                    continue
                cond_committors = tpt.conditional_committors(A, B, C, msm)

                temp = 0.0
                for i in xrange(10):
                    if i in [A, B]:
                        continue
                    temp += cond_committors[i] * msm.transmat_[A, i]
                temp /= denom

                ref_hub_scores[C] += temp

    ref_hub_scores /= (9 * 8)

    print(ref_hub_scores, hub_scores)

    npt.assert_array_almost_equal(ref_hub_scores, hub_scores)
