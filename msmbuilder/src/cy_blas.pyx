# cython: boundscheck=False, cdivision=True, wraparound=False
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Direct BLAS DGEMM (matrix multiply) calls from cython.

See the discussion on scipy-list about this pattern.
http://comments.gmane.org/gmane.comp.python.scientific.devel/19041

Furthermore, since I like to (1) deal with c-major arrays in memory,
and (2) can't remember the DGEMM call signature, this module provides
copy-free wrappers, following from Christoph Lassner's blog post:

  http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
"""

import numpy as np
from scipy.linalg import blas
cimport cython
#from libc.stdio import stderr

cdef extern from "f2py/f2pyptr.h":
    void *f2py_pointer(object) except NULL

ctypedef double d
ctypedef int dgemm_t(char *transa, char *transb, int *m, int *n, int *k, d *alpha, const d *a,
                     int *lda, const d *b, int *ldb, d *beta, d *c, int *ldc) nogil
ctypedef int dgemv_t(char *transa, int *m, int *n, d *alpha, const d *a,
                     int *lda, const d *x, int *incx, d *beta, d *y, int *incy) nogil
ctypedef d ddot_t(int *n, const d *dx, int *incx, const d *dy, int *incy) nogil
ctypedef d dnrm2_t(int *n, const d *x, int *incx) nogil
ctypedef int daxpy_t(int *n, const d *alpha, const d *x, int *incx, d *y, int *incy) nogil
ctypedef int dger_t(int *n, int *m, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil

cdef dgemm_t *FORTRAN_DGEMM = <dgemm_t*>f2py_pointer(blas.dgemm._cpointer)
cdef dgemv_t *FORTRAN_DGEMV = <dgemv_t*>f2py_pointer(blas.dgemv._cpointer)
cdef ddot_t  *FORTRAN_DDOT  = <ddot_t*> f2py_pointer(blas.ddot._cpointer)
cdef dnrm2_t *FORTRAN_DNRM2 = <dnrm2_t*>f2py_pointer(blas.dnrm2._cpointer)
cdef daxpy_t *FORTRAN_DAXPY = <daxpy_t*>f2py_pointer(blas.daxpy._cpointer)
cdef dger_t  *FORTRAN_DGER  = <dger_t*>f2py_pointer(blas.dger._cpointer)


@cython.boundscheck(False)
cdef inline int cdgemm_NN(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) nogil:
    """c = beta*c + alpha*dot(a, b)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.shape[1] != b.shape[0] or a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
        return -1

    FORTRAN_DGEMM("N", "N", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &k, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef inline int cdgemm_NT(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) nogil:
    """c = beta*c + alpha*dot(a, b.T)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[0]
    if a.shape[1] != b.shape[1] or a.shape[0] != c.shape[0] or b.shape[0] != c.shape[1]:
        return -1

    FORTRAN_DGEMM("T", "N", &n, &m, &k, &alpha, &b[0,0], &k, &a[0,0], &k, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef inline int cdgemm_TN(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) nogil:
    """c = beta*c + alpha*dot(a.T, b)
    """
    cdef int m, k, n
    m = a.shape[1]
    k = a.shape[0]
    n = b.shape[1]
    if a.shape[0] != b.shape[0] or a.shape[1] != c.shape[0] or b.shape[1] != c.shape[1]:
        return -1

    FORTRAN_DGEMM("N", "T", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &m, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef int cdgemv_N(const double[:, ::1] a, const double[:] x, double[:] y, double alpha=1.0, double beta=0.0) nogil:
    cdef int m, n, incx, incy
    m = a.shape[0]
    n = a.shape[1]
    incx = x.strides[0] / sizeof(double)
    incy = y.strides[0] / sizeof(double)
    if a.shape[1] != x.shape[0] or a.shape[0] != y.shape[0]:
        return -1

    FORTRAN_DGEMV("T", &n, &m, &alpha, &a[0,0], &n, &x[0], &incx, &beta, &y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdgemv_T(const double[:, ::1] a, const double[:] x, double[:] y, double alpha=1.0, double beta=0.0) nogil:
    cdef int m, n, incx, incy
    m = a.shape[1]
    n = a.shape[0]
    incx = x.strides[0] / sizeof(double)
    incy = y.strides[0] / sizeof(double)
    if a.shape[0] != x.shape[0] or a.shape[1] != y.shape[0]:
        return -1

    FORTRAN_DGEMV("N", &m, &n, &alpha, &a[0,0], &m, &x[0], &incx, &beta, &y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cddot(const double[:] x, const double[:] y, double *result) nogil:
    cdef int n, incx, incy
    n = x.shape[0]
    incx = x.strides[0] / sizeof(double)
    incy = y.strides[0] / sizeof(double)
    if y.shape[0] != n:
        return -1

    result[0] = FORTRAN_DDOT(&n, &x[0], &incx, &y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdnrm2(double[:] x, double* result) nogil:
    cdef int n = x.shape[0]
    cdef int incx = x.strides[0] / sizeof(double)
    result[0] = FORTRAN_DNRM2(&n, &x[0], &incx)
    return 0


@cython.boundscheck(False)
cdef int cdaxpy(double alpha, const double[:] x, double[:] y) nogil:
    cdef int n = x.shape[0]
    cdef int incx = x.strides[0] / sizeof(double)
    cdef int incy = y.strides[0] / sizeof(double)
    FORTRAN_DAXPY(&n, &alpha, &x[0], &incx, &y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdger(double alpha, double[:] x, double[:] y, double[::1, :] A) nogil:
    cdef int n, m, incx, incy, lda
    m = x.shape[0]
    n = y.shape[0]
    incx = x.strides[0] / sizeof(double)
    incy = y.strides[0] / sizeof(double)
    # dger_t(int *n, int *m, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda)
    FORTRAN_DGER(&m, &n, &alpha, &x[0], &incx, &y[0], &incy, &A[0,0], &m)
    return 0
