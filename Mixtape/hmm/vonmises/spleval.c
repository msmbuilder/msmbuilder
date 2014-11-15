/* -------------------------------------------------------------------------- */
/* This is part of the OpenMM molecular simulation toolkit originating from   */
/* Simbios, the NIH National Center for Physics-Based Simulation of           */
/* Biological Structures at Stanford, funded under the NIH Roadmap for        */
/* Medical Research, grant U54 GM072970. See https://simtk.org.               */
/*                                                                            */
/* Portions copyright (c) 2010 Stanford University and the Authors.           */
/* Authors: Peter Eastman                                                     */
/* Contributors:                                                              */
/*                                                                            */
/* Permission is hereby granted, free of charge, to any person obtaining a    */
/* copy of this software and associated documentation files (the "Software"), */
/* to deal in the Software without restriction, including without limitation  */
/* the rights to use, copy, modify, merge, publish, distribute, sublicense,   */
/* and/or sell copies of the Software, and to permit persons to whom the      */
/* Software is furnished to do so, subject to the following conditions:       */
/*                                                                            */
/* The above copyright notice and this permission notice shall be included in */
/* all copies or substantial portions of the Software.                        */
/*                                                                            */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    */
/* THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    */
/* DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      */
/* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  */
/* USE OR OTHER DEALINGS IN THE SOFTWARE.                                     */
/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double evaluateSpline(const double* x, const double* y, const double* deriv, long n, double t) {
  if (t < x[0] || t > x[n-1]) {
    fprintf(stderr, "evaluateSpline: specified point is outside the range defined by the spline\n");
    exit(1);
  }
  
  // Perform a binary search to identify the interval containing the point to evaluate. */
  int lower = 0;
  int upper = n-1;
  while (upper-lower > 1) {
    int middle = (upper+lower)/2;
    if (x[middle] > t)
      upper = middle;
    else
      lower = middle;
  }

  // Evaluate the spline. */
  double dx = x[upper]-x[lower];
  double a = (x[upper]-t)/dx;
  double b = 1.0-a;
  return a*y[lower]+b*y[upper]+((a*a*a-a)*deriv[lower] + (b*b*b-b)*deriv[upper])*dx*dx/6.0;
} 
