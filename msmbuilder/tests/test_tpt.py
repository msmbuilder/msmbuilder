from __future__ import print_function, division, absolute_import

import numpy as np
import numpy.testing as npt

from msmbuilder import tpt
from msmbuilder.msm import MarkovStateModel, BayesianMarkovStateModel


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
                 [0, 1, 3, 5],
                 [0, 1, 5]]

    ref_fluxes = np.array([0.5, 0.3, 0.2])

    res_bottle = tpt.paths(sources, sinks, net_flux, remove_path='bottleneck')
    res_subtract = tpt.paths(sources, sinks, net_flux, remove_path='subtract')

    for paths, fluxes in [res_bottle, res_subtract]:
        npt.assert_array_almost_equal(fluxes, ref_fluxes)
        assert len(paths) == len(ref_paths)

        for i in range(len(paths)):
            npt.assert_array_equal(paths[i], ref_paths[i])


def test_committors_1():
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

    # print(committors, ref)

    npt.assert_array_almost_equal(ref, committors)


def test_committors_2():
    bmsm = BayesianMarkovStateModel(lag_time=1)
    assignments = np.random.randint(3, size=(10, 1000))
    bmsm.fit(assignments)

    committors = tpt.committors([0], [2], bmsm)

    ref = 0
    for tprob in bmsm.all_transmats_:
        ref += np.power(tprob[1, 1], np.arange(1000)).sum() * tprob[1, 2]
    ref = np.array([0, ref / 100., 1])

    npt.assert_array_almost_equal(ref, committors, decimal=2)


def test_cond_committors_1():
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
    ref = (for_committors[1] -
           np.power(tprob[1, 1], np.arange(5000)).sum() *
           tprob[1, 3])
    ref = [0, ref, for_committors[2], 0]

    npt.assert_array_almost_equal(ref, cond_committors)


def test_cond_committors_2():
    # depends on tpt.committors

    bmsm = BayesianMarkovStateModel(lag_time=1)
    assignments = np.random.randint(4, size=(10, 1000))
    bmsm.fit(assignments)

    for_committors = tpt.committors(0, 3, bmsm)
    cond_committors = tpt.conditional_committors(0, 3, 2, bmsm)

    ref = 0
    for tprob in bmsm.all_transmats_:
        ref += (for_committors[1] -
                np.power(tprob[1, 1], np.arange(5000)).sum() *
                tprob[1, 3])
    ref = [0, ref / 100., for_committors[2], 0]

    npt.assert_array_almost_equal(ref, cond_committors, decimal=2)


def test_fluxes_1():
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
    for i in range(3):
        for j in range(3):
            if i != j:
                # Eq. 2.24 in Metzner et al. Transition Path Theory.
                # Multiscale Model. Simul. 2009, 7, 1192-1219.
                ref_fluxes[i, j] = (pop[i] * tprob[i, j] *
                                    (1 - qplus[i]) * qplus[j])

    for i in range(3):
        for j in range(3):
            ref_net_fluxes[i, j] = np.max([0, ref_fluxes[i, j] -
                                          ref_fluxes[j, i]])

    fluxes = tpt.fluxes(0, 2, msm)
    net_fluxes = tpt.net_fluxes(0, 2, msm)

    npt.assert_array_almost_equal(ref_fluxes, fluxes)
    npt.assert_array_almost_equal(ref_net_fluxes, net_fluxes)


def test_fluxes_2():
    # depends on tpt.committors

    bmsm = BayesianMarkovStateModel(lag_time=1)
    assignments = np.random.randint(3, size=(10, 1000))
    bmsm.fit(assignments)

    # forward committors
    qplus = tpt.committors(0, 2, bmsm)

    ref_fluxes = np.zeros((3, 3))
    ref_net_fluxes = np.zeros((3, 3))
    for el in zip(bmsm.all_populations_, bmsm.all_transmats_):
        pop = el[0]
        tprob = el[1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Eq. 2.24 in Metzner et al. Transition Path Theory.
                    # Multiscale Model. Simul. 2009, 7, 1192-1219.
                    ref_fluxes[i, j] += (pop[i] * tprob[i, j] *
                                         (1 - qplus[i]) * qplus[j])

    ref_fluxes /= 100.

    for i in range(3):
        for j in range(3):
            ref_net_fluxes[i, j] = np.max([0, ref_fluxes[i, j] -
                                          ref_fluxes[j, i]])

    fluxes = tpt.fluxes(0, 2, bmsm)
    net_fluxes = tpt.net_fluxes(0, 2, bmsm)

    npt.assert_array_almost_equal(ref_fluxes, fluxes, decimal=2)
    npt.assert_array_almost_equal(ref_net_fluxes, net_fluxes, decimal=2)


def test_hubscore():
    # Make an actual hub!

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
    for A in range(10):
        for B in range(10):
            committors = tpt.committors(A, B, msm)
            denom = msm.transmat_[A, :].dot(committors)
            for C in range(10):
                if A == B or A == C or B == C:
                    continue
                cond_committors = tpt.conditional_committors(A, B, C, msm)

                temp = 0.0
                for i in range(10):
                    if i in [A, B]:
                        continue
                    temp += cond_committors[i] * msm.transmat_[A, i]
                temp /= denom

                ref_hub_scores[C] += temp

    ref_hub_scores /= (9 * 8)

    npt.assert_array_almost_equal(ref_hub_scores, hub_scores)


def test_mfpt_match():
    assignments = np.random.randint(10, size=(10, 2000))
    msm = MarkovStateModel(lag_time=1)
    msm.fit(assignments)

    # these two do different things
    mfpts0 = np.vstack([tpt.mfpts(msm, i) for i in range(10)]).T
    mfpts1 = tpt.mfpts(msm)

    npt.assert_array_almost_equal(mfpts0, mfpts1)


def test_mfpt2():
    tprob = np.array([[0.90, 0.10],
                      [0.22, 0.78]])

    pi0 = 1
    pi1 = pi0 * tprob[0, 1] / tprob[1, 0]
    pops = np.array([pi0, pi1]) / (pi0 + pi1)

    msm = MarkovStateModel(lag_time=1)
    msm.transmat_ = tprob
    msm.n_states_ = 2
    msm.populations_ = pops

    mfpts = np.vstack([tpt.mfpts(msm, i) for i in range(2)]).T

    # since it's a 2x2 the mfpt from 0 -> 1 is the
    # same as the escape time of 0
    npt.assert_almost_equal(1 / (1 - tprob[0, 0]), mfpts[0, 1])
    npt.assert_almost_equal(1 / (1 - tprob[1, 1]), mfpts[1, 0])
