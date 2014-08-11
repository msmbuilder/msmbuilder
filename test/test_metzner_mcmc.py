import numpy as np
from mixtape.markovstatemodel import BayesianMarkovStateModel
from mixtape.markovstatemodel._metzner_mcmc import (metzner_mcmc_fast,
                                                    metzner_mcmc_slow)

def test_1():
    Z = np.array([[1, 10, 2], [2, 26, 3], [15, 20, 20]]).astype(np.double)
    value1 = list(metzner_mcmc_fast(Z, 4, n_thin=1, random_state=0))
    value2 = list(metzner_mcmc_slow(Z, 4, n_thin=1, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value1), np.array(value2))

    value3 = list(metzner_mcmc_fast(Z, 4, n_thin=2, random_state=0))
    value4 = list(metzner_mcmc_slow(Z, 4, n_thin=2, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value3), np.array(value4))
    np.testing.assert_array_almost_equal(np.array(value1)[1::2], np.array(value3))


def test_2():
    Z = np.array([[  5.,   2.], [  1., 10.]])
    value1 = list(metzner_mcmc_fast(Z, 100, n_thin=1, random_state=0))
    value2 = list(metzner_mcmc_slow(Z, 100, n_thin=1, random_state=0))
    np.testing.assert_array_almost_equal(np.array(value1), np.array(value2))
    assert np.all(np.array(value1) > 0)


def test_3():
    trajectory = [0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1]
    msm1 = BayesianMarkovStateModel(sampler='metzner', n_thin=1, n_samples=10, random_state=0)
    msm1.fit([trajectory])
    msm2 = BayesianMarkovStateModel(sampler='metzner_py', n_thin=1, n_samples=10, random_state=0)
    msm2.fit([trajectory])

    np.testing.assert_array_almost_equal(
        msm1.transmat_samples_,
        msm2.transmat_samples_)