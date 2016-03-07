import numpy as np
import scipy.optimize
from msmbuilder.msm._markovstatemodel import _transmat_mle_prinz

random = np.random.RandomState(0)


def reference_mle(Counts, NumIter=10000000, TerminationEpsilon=1E-10):
    # "Boxer method", from Equation (A4) in Bowman et al.
    # J. Chem. Phys. 131, 124101 2009g
    if not scipy.sparse.isspmatrix(Counts):
        Counts = scipy.sparse.csr_matrix(Counts)
    Counts = Counts.asformat("csr").asfptype()
    Counts.eliminate_zeros()

    S = Counts + Counts.transpose()
    N = np.array(Counts.sum(1)).flatten()
    Na = N

    NS = np.array(S.sum(1)).flatten()
    NS /= NS.sum()
    Ind = np.argmax(NS)

    NZX, NZY = np.array(S.nonzero())

    Q = S.copy()
    XS = np.array(Q.sum(0)).flatten()

    for k in range(NumIter):
        Old = XS
        V = Na / XS
        Q.data[:] = S.data / (V[NZX] + V[NZY])
        QS = np.array(Q.sum(0)).flatten()

        XS = QS
        XS /= XS.sum()
        PiDiffNorm = np.linalg.norm(XS - Old)
        if PiDiffNorm < TerminationEpsilon:
            break

    Q = np.array(Q.todense())
    T = (Q.T / Q.sum(axis=1)).T
    return T


def _test(C):
    n = C.shape[0]

    T1, pi1 = _transmat_mle_prinz(C, tol=1e-10)
    T2 = reference_mle(scipy.sparse.csr_matrix(C))

    # reference pi
    eigvals, eigvecs = np.linalg.eig(T1.T)
    pi = eigvecs[:, np.argmax(eigvals)]
    pi = pi / np.sum(pi)

    np.testing.assert_array_almost_equal(T1, T2)
    np.testing.assert_array_almost_equal(pi1, pi)

    # make sure that T1 is reversible
    for i in range(n):
        for j in range(n):
            np.testing.assert_almost_equal(T1[i, j] * pi[i], T1[j, i] * pi[j])


def test_1():
    C = np.array([
        [5.0, 0.0, 3.0, 3.0, 7.0],
        [9.0, 3.0, 5.0, 2.0, 4.0],
        [7.0, 6.0, 8.0, 8.0, 1.0],
        [6.0, 7.0, 7.0, 8.0, 1.0],
        [5.0, 9.0, 8.0, 9.0, 4.0]])

    _test(C)


def test_with_zero():
    # test with a zero in the symmetric count matrix
    C = np.array([
        [5.0, 0.0, 3.0],
        [0.0, 3.0, 5.0],
        [7.0, 6.0, 8.0]])

    _test(C)


def test_with_zero_diagonal():
    # test with a zero on the diagonal
    C = np.array([
        [0.0, 0.0, 3.0],
        [0.0, 3.0, 5.0],
        [7.0, 6.0, 8.0]])

    _test(C)


def test_tolerance():
    with np.testing.assert_raises(ValueError):
        _transmat_mle_prinz(np.zeros((3, 3)), tol=1e-10)
    with np.testing.assert_raises(ValueError):
        _transmat_mle_prinz(-1 * np.ones((3, 3)), tol=1e-10)


def test_floats():
    C = np.array([[0, 1], [1, 0]], dtype=float)
    transmat, populations = _transmat_mle_prinz(C)
    assert np.all(np.isfinite(transmat))
    assert np.all(np.isfinite(populations))
    np.testing.assert_array_equal(transmat, C)


def test_one_state():
    C = np.array([[1]], dtype=float)
    T, pi = _transmat_mle_prinz(C)
    np.testing.assert_array_equal(T, C)


def test_counts_factor():
    C = np.random.randint(10, size=(5, 5)).astype(float)
    transmat1, pi2 = _transmat_mle_prinz(C)

    transmat2, pi1 = _transmat_mle_prinz(10 * C)
    np.testing.assert_array_almost_equal(transmat1, transmat2)
    np.testing.assert_array_almost_equal(pi1, pi2)
