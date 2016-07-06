#!python
#cython: language_level=3, wraparound=False, infer_types=True

# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

cdef extern from "math.h":
    double abs(double m)

cimport cython


cdef inline double sign(double x):
  if x > 0.0:
    return 1.
  elif x < 0.0:
    return -1.
  return 0.0


@cython.boundscheck(False)
def mad_estimate(double[:, ::1] arr,
                 double[::1] mad,
                 double[::1] med,
                 double eta):
    cdef unsigned int i
    cdef unsigned int j
    cdef double dev1, dev2

    cdef unsigned int n_rows = len(arr)
    cdef unsigned int n_cols = len(mad)


    for i in range(n_rows):
        for j in range(n_cols):
          dev1 = arr[i, j] - med[j]
          med[j] += eta * sign(dev1)
          dev2 = abs(arr[i, j] - med[j]) - mad[j]
          mad[j] += eta * sign(dev2)

    return mad, med
