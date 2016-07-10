import numpy as np
from msmbuilder.msm._metzner_mcmc_fast import metzner_mcmc_fast

from msmbuilder.cluster import NDGrid
from msmbuilder.example_datasets import DoubleWell
from msmbuilder.msm import BayesianMarkovStateModel
from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm._metzner_mcmc_slow import metzner_mcmc_slow


def test_1():
    Z = np.array([[1, 10, 2], [2, 26, 3], [15, 20, 20]]).astype(np.double)
    value1 = list(metzner_mcmc_fast(Z, 4, n_thin=1, random_state=0))
    value2 = list(metzner_mcmc_slow(Z, 4, n_thin=1, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value1), np.array(value2))

    value3 = list(metzner_mcmc_fast(Z, 4, n_thin=2, random_state=0))
    value4 = list(metzner_mcmc_slow(Z, 4, n_thin=2, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value3), np.array(value4))
    np.testing.assert_array_almost_equal(
        np.array(value1)[1::2], np.array(value3))


def test_2():
    Z = np.array([[5., 2.], [1., 10.]])
    value1 = list(metzner_mcmc_fast(Z, 100, n_thin=1, random_state=0))
    value2 = list(metzner_mcmc_slow(Z, 100, n_thin=1, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value1), np.array(value2))
    assert np.all(np.array(value1) > 0)


def test_3():
    trajectory = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0]
    msm1 = BayesianMarkovStateModel(sampler='metzner', n_steps=1, n_samples=100,
                                    n_chains=1, random_state=0)
    msm1.fit([trajectory])
    msm2 = BayesianMarkovStateModel(sampler='metzner_py', n_steps=1,
                                    n_samples=100, n_chains=1, random_state=0)
    msm2.fit([trajectory])

    np.testing.assert_array_almost_equal(
        msm1.all_transmats_,
        msm2.all_transmats_)

    assert msm1.all_timescales_.shape == (100, 2)
    assert msm1.all_eigenvalues_.shape == (100, 3)
    assert msm1.all_left_eigenvectors_.shape == (100, 3, 3)
    assert msm1.all_right_eigenvectors_.shape == (100, 3, 3)
    assert msm1.all_populations_.shape == (100, 3)
    np.testing.assert_array_almost_equal(msm1.all_populations_.sum(axis=1),
                                         np.ones(100))


def test_4():
    trajectory = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0]
    msm1 = BayesianMarkovStateModel(n_steps=3, n_samples=10, n_chains=1,
                                    random_state=0).fit([trajectory])
    assert msm1.all_transmats_.shape[0] == 10

    msm2 = BayesianMarkovStateModel(n_steps=4, n_samples=10, n_chains=3,
                                    random_state=0).fit([trajectory])
    assert msm2.all_transmats_.shape[0] == 10


def test_5():
    trjs = DoubleWell(random_state=0).get_cached().trajectories
    clusterer = NDGrid(n_bins_per_feature=5)
    mle_msm = MarkovStateModel(lag_time=100, verbose=False)
    b_msm = BayesianMarkovStateModel(lag_time=100, n_samples=1000, n_chains=8,
                                     n_steps=1000, random_state=0)

    states = clusterer.fit_transform(trjs)
    b_msm.fit(states)
    mle_msm.fit(states)

    # this is a pretty silly test. it checks that the mean transition
    # matrix is not so dissimilar from the MLE transition matrix.
    # This shouldn't necessarily be the case anyways -- the likelihood is
    # not "symmetric". And the cutoff chosen is just heuristic.
    assert np.linalg.norm(b_msm.all_transmats_.mean(axis=0)
                          - mle_msm.transmat_) < 1e-2
