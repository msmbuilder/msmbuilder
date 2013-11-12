import scipy.optimize
import scipy.misc
import numpy as np
from mixtape import _hmm

def test_logsymsumexp():  
    x = (1+np.ones(9)).reshape(3,3)
    x = x + x.T
    xx = np.zeros(3*4/2)
    k = 0
    for i in range(3):
        for j in range(i, 3):
            xx[k] = x[i,j]
            k += 1

    r1 = _hmm.logsymsumexp(xx, 3)
    r2 = scipy.misc.logsumexp(x, axis=0)
    r3 = scipy.misc.logsumexp(x, axis=1)
    np.testing.assert_array_almost_equal(r1, r2)
    np.testing.assert_array_almost_equal(r1, r3)


def test_gradients():
    n_states = 5
    u0 = np.random.rand((n_states*(n_states+1))/2)
    symcounts = np.random.rand((n_states*(n_states+1))/2)
    rowsums = np.random.rand(n_states)
    logrowsums = np.log(rowsums)
    
    error = scipy.optimize.check_grad(_hmm.reversible_transmat_likelihood,
                    _hmm.reversible_transmat_grad,
                    u0, symcounts, rowsums, logrowsums)
    assert error < 1e-5