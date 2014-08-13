import numpy as np
from sklearn.utils import check_random_state
from cython.parallel import prange
cimport cython

cdef extern:
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
        Number of parallel MCMC chains to run.
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


def metzner_mcmc_slow(Z, n_samples, n_thin=1, random_state=None):
    """Metropolis Markov chain Monte Carlo sampler for reversible transition
    matrices

    Parameters
    ----------
    Z : np.array, shape=(n_states, n_states)
        The effective count matrix, the number of observed transitions
        between states plus the number of prior counts
    n_samples : int
        Number of steps to iterate the chain for
    n_thin : int
        Yield every ``n_thin``-th sample from the MCMC chain
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
    metzner_mcmc_fast
    """
    K_MINUS = 0.9
    K_PLUS = 1.1

    Z = np.asarray(Z)
    n_states = Z.shape[0]
    if not Z.ndim == 2 and Z.shape[1] == n_states:
        raise ValueError("Z must be square. Z.shape=%s" % str(Z.shape))
    K = 0.5 * (Z + Z.T) / np.sum(Z, dtype=float)
    random = check_random_state(random_state)
    n_accept = 0

    for t in range(n_samples):
        # proposal
        i, j = (random.rand(2) * n_states).astype(np.int)
        
        sc = np.sum(K)
        if i == j:
            a, b = max(-K[i,j], K_MINUS - sc), K_PLUS - sc
        else:
            a, b = max(-K[i,j], 0.5*(K_MINUS - sc)), 0.5*(K_PLUS - sc)

        epsilon = random.uniform(a, b)
        K_proposal = np.copy(K)
        K_proposal[i, j] += epsilon
        if i != j:
            K_proposal[j, i] += epsilon

        # acceptance?
        cutoff = np.exp(_logprob_T(_K_to_T(K_proposal), Z) -
                        _logprob_T(_K_to_T(K), Z))
        r = random.rand()
        # print 'i', i, 'j', j
        # print 'a', a, 'b', b
        # print 'cutoff', cutoff
        # print 'r', r
        # print 'sc', sc

        if r < cutoff:
            n_accept += 1
            K = K_proposal

        if (t+1) % n_thin == 0:
            yield _K_to_T(K)



def _K_to_T(K):
    return K / np.sum(K, dtype=float, axis=1, keepdims=True)

def _logprob_T(T, Z):
    assert np.all(T > 0)
    return np.sum(np.multiply(Z, np.log(T)))


def _metzner_figure_4():
    """Generate figure 4 from Metzner's paper [1].
    
    This can be used as a rough test of the sampler
    """
    import matplotlib.pyplot as pp
    def _scatter(Ts, xi, xj, yi, yj):
        pp.grid(False)
        pp.hexbin(Ts[:, xi, xj], Ts[:, yi, yj], cmap='hot_r', vmin=0, vmax=100)
        pp.xlabel('T_{%d,%d}' % (xi+1, xj+1))
        pp.ylabel('T_{%d,%d}' % (yi+1, yj+1))
        pp.plot([0,1], [1,0], c='k')
        pp.ylim(0, 1)
        pp.xlim(0, 1)
    
    C = np.array([[1, 10, 2], [2, 26, 3], [15, 20, 20]])
    Ts = np.array(list(metzner_mcmc_slow(C, 100000)))

    pp.figure(figsize=(6, 6)); pp.subplot(axisbg=(0,0,0,0))
    _scatter(Ts, 0, 1, 0, 2)

    pp.figure(figsize=(6, 6)); pp.subplot(axisbg=(0,0,0,0))
    _scatter(Ts, 1, 0, 1, 2)
    pp.figure(figsize=(6, 6)); pp.subplot(axisbg=(0,0,0,0))
    _scatter(Ts, 2, 0, 2, 1)
    pp.show()
