import numpy as np
from sklearn.utils import check_random_state
from cython.parallel import prange
cimport cython

cdef extern from "metzner_mcmc.h":
    void metzner_mcmc_step(const double* Z, const double* N, double* K,
                           double* Q, const double* random, double* sc,
                           int n_states, int n_steps) nogil

@cython.boundscheck(False)
def metzner_mcmc_fast(Z, int n_samples, int n_thin=1, int n_chains=1,
                      random_state=None):
    """Metropolis Markov chain Monte Carlo sampler for reversible transition
    matrices

    Parameters
    ----------
    Z : np.array, shape=(n_states, n_states)
        The effective count matrix, the number of observed transitions
        between states plus the number of prior counts
    n_samples : int
        Number of steps to iterate each chain for
    n_thin : int
        Yield every ``n_thin``-th sample from the MCMC chain
    n_chains : int
        Number of parallel MCMC chains to run. The "inner iterations" of
        each MCMC chain advancing by ``n_thin`` iterations is run in parallel
        using OpenMP (on supported platforms).
    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

    Notes
    -----
    The transition matrix posterior distribution is ::

        P(T | Z) \propto \Prod_{ij} T_{ij}^{Z_{ij}}

    and constrained to be reversible, such that there exists a \pi s.t. ::

        \pi_i T_{ij} = \pi_j T_{ji}

    Yields
    ------
    T : np.array, shape=(n_states, n_states)
        This generator yields samples from the transition matrix posterior

    References
    ----------
    .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
        Distribution of transition matrices and functions of transition
        matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)

    See Also
    --------
    metzner_mcmc_slow
    """

    cdef int i, tid
    cdef double[:,::1] rand
    cdef double[::1] Q
    cdef double[::1] N
    cdef double[:,::1] K
    cdef double[:,:,::1] threadK
    cdef double[:,::1] threadQ
    cdef double[:,::1] Zcopy = np.array(Z, copy=True, dtype=np.double)
    cdef int n_states = len(Z)
    cdef double sc
    if not len(Z[0]) == n_states:
        raise ValueError("Z must be square.")

    # On random number generation for MCMC. It would be a lot easier to just
    # generate the random numbers inside the step kernel, `metzner_mcmc_step`,
    # but unfortunately the PSRNGs in C are garbage. Also, by using the numpy
    # PSRNG and passing the values in, we can ensure that, if the seeds are
    # syncrhonized, this generator and `metzner_mcmc_slow` will yield _exactly_
    # the same values (when n_chains=1), which is useful for testing.
    random = check_random_state(random_state)

    # K are the indepdent variables we're sampling, a symmetric non-negative 2D
    # "virtual" count matrix.
    K = 0.5 * (np.array(Zcopy) + np.array(Zcopy).T) / np.sum(Zcopy)
    # Q is the row-sums of K
    Q = np.sum(K, axis=1, dtype=float)
    # N is the row-sums of Z
    N = np.sum(Zcopy, axis=1, dtype=float)
    sc = np.sum(K)

    # copies of Q and K, one for each chain. Each thread will have a separate
    # copy of Q and K it updates, and then after n_thin iterations we
    # yield from each thread.
    threadQ = np.repeat(np.array(Q).reshape(1, n_states), n_chains, axis=0)
    threadK = np.repeat(np.array(K).reshape(1, n_states, n_states),
                        n_chains, axis=0)

    for i in range(n_samples / n_thin):
        rand = random.rand(n_chains, 4 * n_thin)

        with nogil:
            for tid in prange(n_chains):
                metzner_mcmc_step(
                    &Zcopy[0,0], &N[0], &threadK[tid, 0,0],
                    &threadQ[tid, 0], &rand[tid, 0],
                    &sc, n_states, n_thin)

        for tid in range(n_chains):
            yield np.array(threadK[tid]) / np.array(threadQ[tid])[:, np.newaxis]
