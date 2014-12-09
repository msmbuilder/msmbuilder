# -------------------------------------------------------------------------- #
# Copyright (c) 2014 Stanford University and the Authors.                    #
# Authors: Peter Eastman                                                     #
# Contributors: Robert T. McGibbon                                           #
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

"""
This file is a script that generates three files,

  data/inv_mbessel_x.dat
  data/inv_mbessel_y.dat
  data/inv_mbessel_deriv.dat

which contain the data for a [natural cubic spline](http://mathworld.wolfram.com/CubicSpline.html)
fit for the inverse of the ratio of two modified bessel functions, which
is required for fitting Von-Mises hidden Markov models.

These files are then #included in a C file. See vmhmm.c:inv_mbessel_ratio
and spleval.c

"""

import os
import numpy as np
import scipy.special
from scipy.sparse.linalg import spsolve
from scipy.sparse import dia_matrix


def createNaturalSpline(x, y):
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y vectors must have same length")
    if n < 2:
        raise ValueError("The length of the input array must be at least 2")
    deriv = np.empty(n, dtype=np.double)
    if n == 2:
        # This is just a straight line.
        deriv[0] = 0
        deriv[1] = 0
        return deriv

    a = np.concatenate(([0],  x[1:-1] - x[:-2],   [0]))
    b = np.concatenate(([1],  2*(x[2:] - x[:-2]), [1]))
    c = np.concatenate(([0],  x[2:]-x[1:-1],      [0]))
    rhs = 6 * (
        np.concatenate(([0],  (y[2:]-y[1:-1])/(x[2:]-x[1:-1]),     [0])) - 
        np.concatenate(([0],  (y[1:-1]-y[:-2]) / (x[1:-1]-x[:-2]), [0])))

    matrix = dia_matrix(([a, b, c], [1, 0, -1]), shape=(n, n)).T
    return spsolve(matrix.tocsc(), rhs)


def write_spline_data():
    """Precompute spline coefficients and save them to data files that
    are #included in the remaining c source code. This is a little devious.
    """
    n_points = 1024
    miny, maxy = 1e-5, 700
    y = np.logspace(np.log10(miny), np.log10(maxy), n_points)
    x = scipy.special.iv(1, y) / scipy.special.iv(0, y)

    # fit the inverse function
    derivs = createNaturalSpline(x, np.log(y))
    
    data = os.path.join(os.path.dirname(__file__), 'data')

    print('Writing inverse bessel ratio spline coefficients')
    np.savetxt(os.path.join(data, 'inv_mbessel_x.dat'), x, newline=',\n')
    np.savetxt(os.path.join(data, 'inv_mbessel_y.dat'), np.log(y), newline=',\n')
    np.savetxt(os.path.join(data, 'inv_mbessel_deriv.dat'), derivs, newline=',\n')


if __name__ == '__main__':
    write_spline_data()