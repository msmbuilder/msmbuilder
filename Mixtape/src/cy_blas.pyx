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

import numpy as np
from scipy.linalg import blas

cdef extern from "f2py/f2pyptr.h":
    void *f2py_pointer(object) except NULL

ctypedef double d
ctypedef int dgemm_t(char *transa, char *transb, int *m, int *n, int *k, d *alpha, d *a,
                     int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil
cdef dgemm_t *FORTRAN_DGEMM = <dgemm_t*>f2py_pointer(blas.dgemm._cpointer)


cpdef cdgemm_NN(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] == c.shape[0]
    assert b.shape[1] == c.shape[1]
    FORTRAN_DGEMM("N", "N", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &k, &beta, &c[0,0], &n)

cpdef cdgemm_NT(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[0]
    assert a.shape[1] == b.shape[1]
    assert a.shape[0] == c.shape[0]
    assert b.shape[0] == c.shape[1]
    FORTRAN_DGEMM("T", "N", &n, &m, &k, &alpha, &b[0,0], &k, &a[0,0], &k, &beta, &c[0,0], &n)

    
cpdef cdgemm_TN(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    cdef int m, k, n
    m = a.shape[1]
    k = a.shape[0]
    n = b.shape[1]
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == c.shape[0]
    assert b.shape[1] == c.shape[1]
    FORTRAN_DGEMM("N", "T", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &m, &beta, &c[0,0], &n)