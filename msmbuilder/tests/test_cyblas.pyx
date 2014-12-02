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
