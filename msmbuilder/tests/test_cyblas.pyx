import numpy as np
include "cy_blas.pyx"


def test_NN_1():
    cdef double[:, ::1] a = np.random.randn(10, 5)
    cdef double[:, ::1] b = np.random.randn(5, 7)
    c_reference = np.dot(a, b)

    c = np.zeros((10, 7))
    cdgemm_NN(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)


def test_NN_2():
    cdef double[:, ::1] a = np.random.randn(7, 10)
    cdef double[:, ::1] b = np.random.randn(10, 5)
    c_reference = np.dot(a, b)

    c = np.zeros((7, 5))
    cdgemm_NN(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)


def test_NT_1():
    cdef double[:, ::1] a = np.random.randn(7, 10)
    cdef double[:, ::1] b = np.random.randn(5, 10)
    c_reference = np.dot(a, b.T)

    c = np.zeros((7, 5))
    cdgemm_NT(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)


def test_NT_2():
    cdef double[:, ::1] a = np.random.randn(10, 3)
    cdef double[:, ::1] b = np.random.randn(7, 3)
    c_reference = np.dot(a, b.T)

    c = np.zeros((10, 7))
    cdgemm_NT(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)


def test_TN_2():
    cdef double[:, ::1] a = np.random.randn(7, 4)
    cdef double[:, ::1] b = np.random.randn(7, 3)
    c_reference = np.dot(a.T, b)

    c = np.zeros((4, 3))
    cdgemm_TN(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)

def test_TN_2():
    cdef double[:, ::1] a = np.random.randn(3, 10)
    cdef double[:, ::1] b = np.random.randn(3, 11)
    c_reference = np.dot(a.T, b)

    c = np.zeros((10, 11))
    cdgemm_TN(a, b, c)
    np.testing.assert_array_almost_equal(c, c_reference)


def test_gemv_N_1():
    A = np.random.randn(4, 5)
    x = np.random.randn(5)
    y = np.zeros(4)
    cdgemv_N(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A, x))


def test_gemv_N_2():
    A = np.random.randn(4, 5)
    x = np.random.randn(50)[::10]
    y = np.zeros(4)
    cdgemv_N(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A, x))


def test_gemv_N_3():
    A = np.random.randn(4, 5)
    x = np.random.randn(50)[::10]
    y = np.zeros(40)[::10]
    cdgemv_N(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A, x))


def test_gemv_T_1():
    A = np.random.randn(50, 4)
    x = np.random.randn(50)
    y = np.zeros(4)
    cdgemv_T(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A.T, x))


def test_gemv_T_2():
    A = np.random.randn(50, 4)
    x = np.random.randn(50*2)[::2]
    y = np.zeros(4)
    cdgemv_T(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A.T, x))


def test_gemv_T_3():
    A = np.random.randn(50, 4)
    x = np.random.randn(50*7)[::7]
    y = np.zeros(4*13)[::13]
    cdgemv_T(A, x, y)
    np.testing.assert_array_almost_equal(y, np.dot(A.T, x))


def test_ddot():
    cdef double result
    x = np.random.randn(50)[::5]
    y = np.random.randn(60)[::6]
    cddot(x, y, &result)
    np.testing.assert_almost_equal(np.dot(x, y), result)


def test_dger_1():
    x = np.random.randn(50)
    y = np.random.randn(50)

    A = np.zeros((50, 50), order='F')
    cdger(1.0, x, y, A)
    np.testing.assert_almost_equal(np.outer(x, y), A)


def test_dger_2():
    x = np.random.randn(5)
    y = np.random.randn(10)
    A = np.zeros((5, 10), order='F')
    result = np.outer(x, y)
    cdger(1.0, x, y, A)
    np.testing.assert_almost_equal(result, A)

def test_dger_3():
    x = np.random.randn(2, 5)
    y = np.random.randn(2, 10)
    A = np.zeros((5, 10), order='F')
    for i in range(2):
        cdger(0.5, x[i], y[i], A)

    np.testing.assert_almost_equal(0.5*np.dot(x.T, y), A)
