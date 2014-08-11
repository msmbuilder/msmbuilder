import numpy as np

cdef extern from "transmat_mle_prinz.h":
    int transmat_mle_prinz(const double* C, int n_states,
                           double tol, double* T, double* pi)

def _transmat_mle_prinz(double[:, ::1] C, double tol=1e-10):
    """Compute a maximum likelihood reversible transition matrix, given
    a set of directed transition counts.

    Algorithim 1 of Prinz et al.[1]

    Parameters
    ----------
    C : (input) 2d array of shape=(n_states, n_states)
        The directed transition counts, in C (row-major) order.
    tol : (input) float
        Convergence tolerance. The algorithm will iterate until the
        change in the log-likelihood is less than `tol`.

    Returns
    -------
    T : (output) pointer to output 2d array of shape=(n_states, n_states)
        Once the algorithim is completed, the resulting transition
        matrix will be written to `T`.
    populations : array, shape = (n_states_,)
        The equilibrium population (stationary left eigenvector) of T

     References
     ----------
     .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    """

    cdef int n_states = len(C)
    if n_states == 0:
        return np.zeros((0, 0)), np.zeros(0)

    if len(C[0]) != n_states:
        raise ValueError('C must be square')
    cdef double[:, ::1] T = np.zeros((n_states, n_states))
    cdef double[::1] pi = np.zeros(n_states)
    cdef int n_iter

    n_iter = transmat_mle_prinz(&C[0,0], n_states, tol, &T[0,0], &pi[0]);
    if n_iter < 0:
        # diagnose the error
        msg = ' Error code=%d' % n_iter
        if np.any(np.less(C, 0)):
            msg = 'Domain error. C must be positive.' + msg
        if np.any(np.sum(C, axis=1) == 0):
            msg = 'Row-sums of C must be positive.' + msg
        raise ValueError(msg)

    return np.array(T), np.array(pi)
