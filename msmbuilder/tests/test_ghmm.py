from __future__ import print_function, division

import warnings
from itertools import permutations

import hmmlearn.hmm
import numpy as np

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import SuperposeFeaturizer
from msmbuilder.hmm import GaussianHMM

rs = np.random.RandomState(42)


def test_ala2():
    # creates a 4-state HMM on the ALA2 data. Nothing fancy, just makes
    # sure the code runs without erroring out
    trajectories = AlanineDipeptide().get_cached().trajectories
    topology = trajectories[0].topology

    indices = topology.select('symbol C or symbol O or symbol N')
    featurizer = SuperposeFeaturizer(indices, trajectories[0][0])

    sequences = featurizer.transform(trajectories)
    hmm = GaussianHMM(n_states=4, n_init=3, random_state=rs)
    hmm.fit(sequences)

    assert len(hmm.timescales_ == 3)
    assert np.any(hmm.timescales_ > 50)


def create_timeseries(means, vars, transmat):
    """Construct a random timeseries based on a specified Markov model."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = hmmlearn.hmm.GaussianHMM(n_components=len(means),
                                         random_state=rs)
        model.means_ = means
        model.covars_ = vars
        model.transmat_ = transmat
        X, Y = model.sample(1000)
    return X


def validate_timeseries(means, vars, transmat, model,
                        valuetol=1e-3, transmattol=1e-3):
    """Whether our model matches the one used to create the timeseries."""
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
            if abs(means[i] - model.means_[order[i]]) > valuetol:
                match = False
                break
            if abs(vars[i] - model.vars_[order[i]]) > valuetol:
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
    means = np.array([[0.0], [5.0]])
    vars = np.array([[1.0], [1.0]])
    X = [create_timeseries(means, vars, transmat) for i in range(10)]

    # For each value of various options,
    # create a 2 state HMM and see if it is correct.
    class two_state_tester(object):
        def __init__(self, init_algo, reversible_type):
            self.init_algo = init_algo
            self.reversible_type = reversible_type
            self.description = ("{}.test_3_state_{}_{}"
                                .format(__name__, init_algo, reversible_type))

        def __call__(self, *args, **kwargs):
            model = GaussianHMM(n_states=2, init_algo=self.init_algo,
                                reversible_type=self.reversible_type,
                                thresh=1e-4, n_iter=30, random_state=rs)
            model.fit(X)
            validate_timeseries(means, vars, transmat, model, 0.1, 0.05)
            assert abs(model.fit_logprob_[-1] - model.score(X)) < 0.5

    for init_algo in ('kmeans', 'GMM'):
        for reversible_type in ('mle', 'transpose'):
            yield two_state_tester(init_algo, reversible_type)


def test_3_state():
    transmat = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.8, 0.2, 0.0]])
    means = np.array([[0.0], [10.0], [5.0]])
    vars = np.array([[1.0], [2.0], [0.3]])
    X = [create_timeseries(means, vars, transmat) for i in range(20)]

    # For each value of various options,
    # create a 3 state HMM and see if it is correct.
    class three_state_tester(object):
        def __init__(self, init_algo, reversible_type):
            self.init_algo = init_algo
            self.reversible_type = reversible_type
            self.description = ("{}.test_2_state_{}_{}"
                                .format(__name__, init_algo, reversible_type))

        def __call__(self, *args, **kwargs):
            model = GaussianHMM(n_states=3, init_algo=self.init_algo,
                                reversible_type=self.reversible_type,
                                thresh=1e-4, n_iter=30, random_state=rs)
            model.fit(X)
            validate_timeseries(means, vars, transmat, model, 0.1, 0.1)
            assert abs(model.fit_logprob_[-1] - model.score(X)) < 0.5

    for init_algo in ('kmeans', 'GMM'):
        for reversible_type in ('mle', 'transpose'):
            yield three_state_tester(init_algo, reversible_type)
