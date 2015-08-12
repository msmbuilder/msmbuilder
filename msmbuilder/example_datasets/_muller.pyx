import numpy as np
from sklearn.utils import check_random_state
from libc.math cimport exp, sqrt
from numpy cimport npy_intp

cdef double *MULLER_aa = [-1, -1, -6.5, 0.7]
cdef double *MULLER_bb = [0, 0, 11, 0.6]
cdef double *MULLER_cc = [-10, -10, -6.5, 0.7]
cdef double *MULLER_AA = [-200, -100, -170, 15]
cdef double *MULLER_XX = [1, 0, -0.5, -1]
cdef double *MULLER_YY = [0, 0.5, 1.5, 1]


def propagate(int n_steps=5000, x0=None, int thin=1,
              double kT=1.5e4, double dt=0.1, double D=0.010,
              random_state=None, double min_x=-np.inf, double max_x=np.inf,
              double min_y=-np.inf, double max_y=np.inf):
    random = check_random_state(random_state)
    if x0 is None:
        x0 = (-0.5, 0.5)
    cdef int i, j
    cdef int save_index = 0
    cdef double DT_SQRT_2D = dt * sqrt(2 * D)
    cdef double beta = 1.0 / kT
    cdef double[:, ::1] r = random.randn(n_steps, 2)
    cdef double[:, ::1] saved_x = np.zeros(((n_steps)/thin, 2))
    cdef double x = x0[0]
    cdef double y = x0[1]
    cdef double[2] grad

    with nogil:
        for i in range(n_steps):
            _muller_grad(x, y, beta, &grad[0])

            # Brownian update
            x = x - dt * grad[0] + DT_SQRT_2D * r[i, 0]
            y = y - dt * grad[1] + DT_SQRT_2D * r[i, 1]

            if x > max_x:
                x = 2 * max_x - x
            if x < min_x:
                x = 2 * min_x - x
            if y > max_y:
                y = 2 * max_y - y
            if y < min_y:
                y = 2 * min_y - y

            if i % thin == 0:
                saved_x[save_index, 0] = x
                saved_x[save_index, 1] = y
                save_index += 1

    return np.asarray(saved_x)


def muller_potential(x, y, beta=1):
    """Muller potential.

    This function can take vectors (x,y) as input and will broadcast
    """
    wrapper = lambda x, y, beta: _muller_potential(x, y, beta)
    return np.vectorize(wrapper)(x, y, beta)


cdef double _muller_potential(double x, double y, double beta=1) nogil:
    cdef npy_intp j
    cdef double value = 0

    for j in range(4):
        value += MULLER_AA[j] * exp(
            MULLER_aa[j] * (x - MULLER_XX[j])**2
            + MULLER_bb[j] * (x - MULLER_XX[j]) * (y - MULLER_YY[j])
            + MULLER_cc[j] * (y - MULLER_YY[j])**2)
    return beta * value


cdef void _muller_grad(double x, double y, double beta, double grad[2]) nogil:
    cdef int j
    cdef double value = 0
    cdef double term
    grad[0] = 0
    grad[1] = 0

    for j in range(4):
        # this is the potential term
        term = MULLER_AA[j] * exp(
            MULLER_aa[j] * (x - MULLER_XX[j])**2
            + MULLER_bb[j] * (x - MULLER_XX[j]) * (y - MULLER_YY[j])
            + MULLER_cc[j] * (y - MULLER_YY[j])**2)

        grad[0] += (2 * MULLER_aa[j] * (x - MULLER_XX[j])
                + MULLER_bb[j] * (y - MULLER_YY[j])) * term
        grad[1] += (MULLER_bb[j] * (x - MULLER_XX[j])
                + 2 * MULLER_cc[j] * (y - MULLER_YY[j])) * term

    grad[0] *= beta
    grad[1] *= beta
