import numpy as np
from sklearn.utils import check_random_state
from libc.math cimport exp, sqrt
cimport cython

cdef extern from "src/muller.c":
    void _muller_grad(const double x[2], double beta, double grad[2]) nogil
    double _muller_potential(double x[2], double beta) nogil


def muller_potential(x, y):
    """Muller potential.
    
    This function can take vectors (x,y) as input and will broadcast
    """
    def wrapper(double x, double y):
        cdef double[:] pos = np.array([x,y])
        return _muller_potential(&pos[0], 1)
    return np.vectorize(wrapper)(x, y)


@cython.boundscheck(False)
@cython.wraparound(False)
def propagate(int n_steps=5000, x0=[-0.5, 0.5], int thin=1,
              double kT=1.5e4, double dt=0.1, double D=0.010,
              random_state=None):
    cdef int i, j
    cdef int save_index = 0
    cdef double DT_SQRT_2D = dt * sqrt(2 * D)
    cdef double beta = 1.0 / kT
    random = check_random_state(random_state)
    cdef double[:, ::1] r = random.randn(n_steps, 2)
    cdef double[:, ::1] saved_x = np.zeros(((n_steps)/thin, 2))
    cdef double x[2]
    cdef double grad[2]

    x[0] = x0[0]
    x[1] = x0[1]

    with nogil:
        for i in range(n_steps):
            _muller_grad(&x[0], beta, &grad[0])
            for j in range(2):
                # this is the key update equation
                x[j] = x[j] - dt * grad[j] + DT_SQRT_2D * r[i, j]
            if i % thin == 0:
                saved_x[save_index, 0] = x[0]
                saved_x[save_index, 1] = x[1]
                save_index += 1

    return np.asarray(saved_x)

