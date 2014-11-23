# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

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