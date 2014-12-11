"""

"""
import numpy as np
import scipy.linalg
from msmbuilder.decomposition.speigh import speigh, scdeflate
random = np.random.RandomState(1)
#np.set_printoptions(precision=3, suppress=True)


def rand_pos_semidef(n):
    # random positive semidefinite matrix
    # http://stackoverflow.com/a/619406/1079728
    # (construct cholesky factor, and then return matrix)
    A = random.rand(n, n)
    B = np.dot(A, A.T)
    return B


def rand_sym(n):
    # random symmetric
    A = random.randn(n, n)
    return A + A.T

def eigh(A, B=None):
    w, V = scipy.linalg.eigh(A, b=B)
    order = np.argsort(-w)
    w = w[order]
    V = V[:, order]
    return w, V

#######################################################


class Test_scdeflate(object):
    def test_1(self):
        n = 4
        A = rand_sym(n)

        w1, V1 = eigh(A)
        Ad = scdeflate(A, V1[:, 0])
        w2, V2 = eigh(Ad)

        self.assert_deflated(w1, V1, w2, V2)

    def test_2(self):
        n = 4
        A = rand_sym(n)
        B = rand_pos_semidef(n)
        w1, V1 = eigh(A, B)

        Ad = scdeflate(A, V1[:, 0])
        w2, V2 = eigh(Ad, B)

        self.assert_deflated(w1, V1, w2, V2)

    def assert_deflated(self, w1, V1, w2, V2):
        # the deflated matrix should have a one zero eigenvalue for the
        # vector that was deflated out.
        near_zero = (np.abs(w2) < 1e-10)
        assert np.sum(near_zero) == 1
        # the other eigenvalues should be unchanged
        np.testing.assert_array_almost_equal(
            w1[1:], w2[np.logical_not(near_zero)])

        remaining_V1 = V1[:, 1:]
        remaining_V2 = V2[:, np.logical_not(near_zero)]
        for i in range(remaining_V2.shape[1]):
            assert (np.allclose(remaining_V1[:, i],  remaining_V2[:, i]) or
                    np.allclose(remaining_V1[:, i], -remaining_V2[:, i]))


class Test_speigh_1(object):
    def test_1(self):
        # test with indefinite A matrix, identity B
        n = 4
        A = rand_sym(n)
        B = np.eye(n)
        w, V = eigh(A, B)
        v_init = V[:, 0] + 0.1*random.randn(n)

        w0, v0, v0f = speigh(A, B, v_init=v_init, rho=0,  return_x_f=True)
        self.assert_close(w0, v0, v0f, A, B)

    def test_2(self):
        # test with indefinite B matrix, indefinite B
        n = 4
        A = rand_sym(n)
        B = rand_pos_semidef(n)
        w, V = eigh(A, B)
        v_init = V[:, 0] + 0.1*random.randn(n)

        w0, v0, v0f = speigh(A, B, v_init=v_init, rho=0, return_x_f=True)
        self.assert_close(w0, v0, v0f, A, B)

    def test_3(self):
        # test with positive semidefinite A matrix, and diagonal
        # matrix B
        n = 4
        A = rand_pos_semidef(n)
        B = np.diag(np.random.randn(n)**2)

        w, V = eigh(A, B)
        v_init = V[:, 0] + 0.1*random.randn(n)

        w0, v0, v0f = speigh(A, B, v_init=v_init, rho=0, return_x_f=True)
        self.assert_close(w0, v0, v0f, A, B)

    def test_4(self):
        # test with positive semidefinite A matrix, and general
        # matrix B
        n = 4
        A = rand_pos_semidef(n)
        B = rand_pos_semidef(n) + np.eye(n)
        w, V = eigh(A, B)
        v_init = V[:, 0] + 0.1*random.randn(n)

        w0, v0, v0f = speigh(A, B, v_init=v_init, rho=0, return_x_f=True)
        self.assert_close(w0, v0, v0f, A, B)

    def assert_close(self, w0, v0, v0f, A, B):
        w, V = eigh(A, B)

        v0 /= np.linalg.norm(v0)
        v0f /= np.linalg.norm(v0f)
        V[:, 0] /= np.linalg.norm(V[:,0])

        np.testing.assert_almost_equal(w0, w[0])
        assert (np.allclose(v0,  V[:, 0]) or
                np.allclose(v0, -V[:, 0]))
        assert (np.linalg.norm(v0f + V[:, 0]) < 1e-3 or
                np.linalg.norm(v0f - V[:, 0]) < 1e-3)


class Test_speigh_2(object):
    def test_1(self):
        # test with indefinite A matrix, identity B
        n = 4
        x = np.array([1.0, 2.0, 3.0, 0])
        A = np.outer(x, x)
        B = np.eye(n)
        w, V = eigh(A, B)

        w0, v0, v0f = speigh(A, B, v_init=np.ones(4), rho=0.0001, verbose=True, return_x_f=True)
        self.assert_close(w0, v0, v0f, A, B)

    def assert_close(self, w0, v0, v0f, A, B):
        w, V = eigh(A, B)

        v0 /= np.linalg.norm(v0)
        v0f /= np.linalg.norm(v0f)
        V[:, 0] /= np.linalg.norm(V[:,0])

        np.testing.assert_almost_equal(w0, w[0])
        assert (np.allclose(v0,  V[:, 0]) or
                np.allclose(v0, -V[:, 0]))
        assert (np.linalg.norm(v0f + V[:, 0]) < 1e-3 or
                np.linalg.norm(v0f - V[:, 0]) < 1e-3)

