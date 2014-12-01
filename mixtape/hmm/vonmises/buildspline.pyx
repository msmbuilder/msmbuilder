# -------------------------------------------------------------------------- #
# This is part of the OpenMM molecular simulation toolkit originating from   #
# Simbios, the NIH National Center for Physics-Based Simulation of           #
# Biological Structures at Stanford, funded under the NIH Roadmap for        #
# Medical Research, grant U54 GM072970. See https://simtk.org.               #
#                                                                            #
# Portions copyright (c) 2010 Stanford University and the Authors.           #
# Authors: Peter Eastman                                                     #
# Contributors:                                                              #
#                                                                            #
# Permission is hereby granted, free of charge, to any person obtaining a    #
# copy of this software and associated documentation files (the "Software"), #
# to deal in the Software without restriction, including without limitation  #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,   #
# and/or sell copies of the Software, and to permit persons to whom the      #
# Software is furnished to do so, subject to the following conditions:       #
#                                                                            #
# The above copyright notice and this permission notice shall be included in #
# all copies or substantial portions of the Software.                        #
#                                                                            #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    #
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  #
# USE OR OTHER DEALINGS IN THE SOFTWARE.                                     #
# -------------------------------------------------------------------------- #


import cython
import numpy as np
cimport numpy as np
np.import_array()

def createNaturalSpline(np.ndarray[ndim=1, dtype=np.double_t] x,
                        np.ndarray[ndim=1, dtype=np.double_t] y):
    cdef int n = len(x)
    if len(y) != n:
        raise ValueError("x and y vectors must have same length")
    if n < 2:
        raise ValueError("The length of the input array must be at least 2")
    cdef np.ndarray[ndim=1, dtype=np.double_t] deriv = np.empty(n, dtype=np.double)
    if n == 2:
        # This is just a straight line.
        deriv[0] = 0
        deriv[1] = 0
        return deriv

    cdef np.ndarray[ndim=1, dtype=np.double_t] a = np.empty(n, dtype=np.double)
    cdef np.ndarray[ndim=1, dtype=np.double_t] b = np.empty(n, dtype=np.double)
    cdef np.ndarray[ndim=1, dtype=np.double_t] c = np.empty(n, dtype=np.double)
    cdef np.ndarray[ndim=1, dtype=np.double_t] rhs = np.empty(n, dtype=np.double)

    a[0] = 0.0
    b[0] = 1.0
    c[0] = 0.0
    rhs[0] = 0.0

    for i in range(1, n-1):
        a[i] = x[i]-x[i-1]
        b[i] = 2.0*(x[i+1]-x[i-1])
        c[i] = x[i+1]-x[i]
        rhs[i] = 6.0*((y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]))
    a[n-1] = 0.0
    b[n-1] = 1.0
    c[n-1] = 0.0
    rhs[n-1] = 0.0
    return solveTridiagonalMatrix(a, b, c, rhs)


cdef solveTridiagonalMatrix(np.ndarray[ndim=1, dtype=np.double_t] a,
                            np.ndarray[ndim=1, dtype=np.double_t] b,
                            np.ndarray[ndim=1, dtype=np.double_t] c,
                            np.ndarray[ndim=1, dtype=np.double_t] rhs):
    cdef int n = len(a)
    cdef double beta
    cdef np.ndarray[ndim=1, dtype=np.double_t] gamma = np.empty(n, dtype=np.double)
    cdef np.ndarray[ndim=1, dtype=np.double_t] solution = np.empty(n, dtype=np.double)

    # Decompose the matrix.
    solution[0] = rhs[0]/b[0]
    beta = b[0]
    for i in range(1, n):
        gamma[i] = c[i-1]/beta
        beta = b[i]-a[i]*gamma[i]
        solution[i] = (rhs[i]-a[i]*solution[i-1])/beta
    # Perform backsubstitation.
    for i in range(n-2, -1, -1):
        solution[i] -= gamma[i+1]*solution[i+1];
    return solution
