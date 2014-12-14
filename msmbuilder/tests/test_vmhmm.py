from __future__ import print_function, division
import sys
import time

import numpy as np
from msmbuilder.hmm import VonMisesHMM
from msmbuilder.hmm.vmhmm import inverse_mbessel_ratio, circwrap
from msmbuilder.hmm import _vmhmm
from sklearn.hmm import GaussianHMM

try:
    from munkres import Munkres
except ImportError:
    print('https://pypi.python.org/pypi/munkres/ for hungarian algorithim',
          file=sys.stderr)
    raise


def test_1():
    vm = VonMisesHMM(n_states=5)
    gm = GaussianHMM(n_components=5)
    X1 = np.random.randn(100, 2)
    yield lambda: vm.fit([X1])
    yield lambda: gm.fit([X1])


def test_3():
    "The inverse function is really the inverse of the forward function"
    np.random.seed(42)
    y = np.random.random(size=100)
    x = inverse_mbessel_ratio(y)
    y2 = inverse_mbessel_ratio.bessel_ratio(x)
    np.testing.assert_array_almost_equal(y, y2, decimal=4)


def test_6():
    # Test that _c_fitkappa is consistent with the two-step python implementation
    np.random.seed(42)
    vm = VonMisesHMM(n_states=13)
    kappas = np.random.randn(13, 7)
    posteriors = np.random.randn(100, 13)
    obs = np.random.randn(100, 7)
    means = np.random.randn(13, 7)

    vm.kappas_ = kappas
    vm._c_fitkappas(posteriors, obs, means)
    c_kappas = np.copy(vm._kappas_)

    vm._py_fitkappas(posteriors, obs, means)
    py_kappas = np.copy(vm._kappas_)
    np.testing.assert_array_almost_equal(py_kappas, c_kappas)


def test_8():
    # "Sample from a VMHMM and then fit to it"
    n_states = 2
    vm = VonMisesHMM(n_states=n_states)
    means = np.array([[0, 0, 0], [np.pi, np.pi, np.pi]])
    kappas = np.array([[1, 2, 3], [2, 3, 4]])
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    vm.means_ = means
    vm.kappas_ = kappas
    vm.transmat_ = transmat
    x, s = vm.sample(1000)

    vm = VonMisesHMM(n_states=2)
    vm.fit([x])

    mappingcost = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            mappingcost[i, j] = np.sum(
                circwrap(vm.means_[i, :] - means[j, :]) ** 2)

    mapping = Munkres().compute(mappingcost)
    means_ = np.array([vm.means_[j, :] for i, j in mapping])
    kappas_ = np.array([vm.kappas_[j, :] for i, j in mapping])
    transmat_ = vm.transmat_

    print('means\n', means, '\nmeans_\n', means_)
    print('kappas\n', kappas, '\nkappas_\n', kappas_)
    print('transmat\n', transmat, '\ntransmat_\n', transmat_)

    #vm.score(x)

    assert np.all(np.abs(kappas - kappas_) < 0.5)
    assert np.all(circwrap(means - means_) < 0.25)


def test_log_likelihood():
    n_samples, n_states, n_features = 1000, 27, 16
    obs = np.random.rand(n_samples, n_features)
    vm = VonMisesHMM(n_states=n_states)
    vm.fit([obs])

    t0 = time.time()
    from scipy.stats.distributions import vonmises

    reference = np.array(
        [np.sum(vonmises.logpdf(obs, vm.kappas_[i], vm.means_[i]), axis=1)
         for i in range(n_states)]).T
    t1 = time.time()
    value = _vmhmm._compute_log_likelihood(obs, vm.means_, vm.kappas_)
    t2 = time.time()

    print("Log likeihood timings")
    print('reference time ', t1 - t0)
    print('c time         ', t2 - t1)
    np.testing.assert_array_almost_equal(reference, value)
