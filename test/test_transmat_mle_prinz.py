import numpy as np
from mdtraj.utils import timing
from mixtape._markovstatemodel import _transmat_mle_prinz
from mixtape._reversibility import reversible_transmat

random = np.random.RandomState(0)

def test_1():
    C = random.randint(10, size=(5,5)).astype(float)
    T1, pi1 = _transmat_mle_prinz(C, tol=1e-10)
    T2, pi2 = reversible_transmat(C)

    # reference pi
    eigvals, eigvecs = np.linalg.eig(T1.T)
    pi = eigvecs[:, np.argmax(eigvals)]
    pi = pi / np.sum(pi)

    np.testing.assert_array_almost_equal(T1, T2)
    np.testing.assert_array_almost_equal(pi1, pi2)
    np.testing.assert_array_almost_equal(pi1, pi)


def test_2():
    # this is just a timing comparison
    C = random.randint(10, size=(200, 200)).astype(float)
    with timing("_transmat_mle_prinz timing"):
        T1, pi1 = _transmat_mle_prinz(C, tol=1e-10)
    with timing("reversible_transmat timing"):
        T2, pi2 = reversible_transmat(C)

    
