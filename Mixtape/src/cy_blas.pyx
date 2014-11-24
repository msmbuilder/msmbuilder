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

cdef extern from "f2py/f2pyptr.h":
    void *f2py_pointer(object) except NULL

ctypedef double d
ctypedef int dgemm_t(char *transa, char *transb, int *m, int *n, int *k, d *alpha, d *a,
                     int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil
cdef dgemm_t *FORTRAN_DGEMM = <dgemm_t*>f2py_pointer(blas.dgemm._cpointer)


cdef inline cdgemm_NN(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    """c = beta*c + alpha*dot(a, b)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] == c.shape[0]
    assert b.shape[1] == c.shape[1]
    FORTRAN_DGEMM("N", "N", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &k, &beta, &c[0,0], &n)

cdef inline cdgemm_NT(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    """c = beta*c + alpha*dot(a, b.T)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[0]
    assert a.shape[1] == b.shape[1]
    assert a.shape[0] == c.shape[0]
    assert b.shape[0] == c.shape[1]
    FORTRAN_DGEMM("T", "N", &n, &m, &k, &alpha, &b[0,0], &k, &a[0,0], &k, &beta, &c[0,0], &n)


cdef inline cdgemm_TN(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0):
    """c = beta*c + alpha*dot(a.T, b)
    """
    cdef int m, k, n
    m = a.shape[1]
    k = a.shape[0]
    n = b.shape[1]
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == c.shape[0]
    assert b.shape[1] == c.shape[1]
    FORTRAN_DGEMM("N", "T", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &m, &beta, &c[0,0], &n)
