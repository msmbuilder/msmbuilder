cimport numpy as np

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,        # Length of vectors
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int idamax_t(
    # IDAMAX finds the index of element having max. absolute value.
    int *n,        # length of vector x
    double *x,     # Vector x
    int * incx     # The increment between elements of x (usually 1)
)

ctypedef int isamax_t(
    # ISAMAX finds the index of element having max. absolute value.
    int *n,              # length of vector x
    np.float32_t *x,     # Vector x
    int * incx           # The increment between elements of x (usually 1)
)
