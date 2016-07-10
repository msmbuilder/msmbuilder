from __future__ import print_function, division

import random
from itertools import permutations

import numpy as np
from scipy.stats.distributions import vonmises

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.hmm import VonMisesHMM


def test_code_works():
    # creates a 4-state HMM on the ALA2 data. Nothing fancy, just makes
    # sure the code runs without erroring out
    trajectories = AlanineDipeptide().get_cached().trajectories
    topology = trajectories[0].topology

    indices = topology.select('symbol C or symbol O or symbol N')
    featurizer = DihedralFeaturizer(['phi', 'psi'], trajectories[0][0])

    sequences = featurizer.transform(trajectories)

    hmm = VonMisesHMM(n_states=4, n_init=1)
    hmm.fit(sequences)

    assert len(hmm.timescales_ == 3)
    assert np.any(hmm.timescales_ > 50)


def circwrap(x):
    """Wrap an array on (-pi, pi)"""
    return x - 2 * np.pi * np.floor(x / (2 * np.pi) + 0.5)


def create_timeseries(means, kappas, transmat):
    """Construct a random timeseries based on a specified Markov model."""
    numStates = len(means)
    state = random.randint(0, numStates - 1)
    cdf = np.cumsum(transmat, 1)
    numFrames = 1000
    X = np.empty((numFrames, 1))
    for i in range(numFrames):
        rand = random.random()
        state = (cdf[state] > rand).argmax()
        X[i, 0] = circwrap(vonmises.rvs(kappas[state], means[state]))
    return X


def validate_timeseries(means, kappas, transmat, model, meantol,
                        kappatol, transmattol):
    """Test our model matches the one used to create the timeseries."""
    numStates = len(means)
    assert len(model.means_) == numStates
    assert (model.transmat_ >= 0.0).all()
    assert (model.transmat_ <= 1.0).all()
    totalProbability = sum(model.transmat_.T)
    assert (abs(totalProbability - 1.0) < 1e-5).all()

    # The states may have come out in a different order,
    # so we need to test all possible permutations.

    for order in permutations(range(len(means))):
        match = True
        for i in range(numStates):
            if abs(circwrap(means[i] - model.means_[order[i]])) > meantol:
                match = False
                break
            if abs(kappas[i] - model.kappas_[order[i]]) > kappatol:
                match = False
                break
            for j in range(numStates):
                diff = transmat[i, j] - model.transmat_[order[i], order[j]]
                if abs(diff) > transmattol:
                    match = False
                    break
        if match:
            # It matches.
            return

    # No permutation matched.
    assert False


def test_2_state():
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
    means = np.array([[0.0], [2.0]])
    kappas = np.array([[4.0], [8.0]])
    X = [create_timeseries(means, kappas, transmat) for i in range(10)]

    # For each value of various options,
    # create a 2 state HMM and see if it is correct.

    for reversible_type in ('mle', 'transpose'):
        model = VonMisesHMM(n_states=2, reversible_type=reversible_type,
                            thresh=1e-4, n_iter=30)
        model.fit(X)
        validate_timeseries(means, kappas, transmat, model, 0.1, 0.5, 0.05)
        assert abs(model.fit_logprob_[-1] - model.score(X)) < 0.5


def test_3_state():
    transmat = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.8, 0.2, 0.0]])
    means = np.array([[0.0], [2.0], [4.0]])
    kappas = np.array([[8.0], [8.0], [6.0]])
    X = [create_timeseries(means, kappas, transmat) for i in range(20)]

    # For each value of various options,
    # create a 3 state HMM and see if it is correct.

    for reversible_type in ('mle', 'transpose'):
        model = VonMisesHMM(n_states=3, reversible_type=reversible_type,
                            thresh=1e-4, n_iter=30)
        model.fit(X)
        validate_timeseries(means, kappas, transmat, model, 0.1, 0.5, 0.1)
        assert abs(model.fit_logprob_[-1] - model.score(X)) < 0.5
