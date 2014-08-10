import numpy as np
from sklearn.utils import check_random_state


def metzner_mcmc_fast(Z, n_samples, n_thin=1, random_state=None):
    # Prepare the variables
    triu = np.triu_indices_from(Z)

    # S is Z + Z.T in symmetric storage without double counting the diagonal
    S = (Z + Z.T - np.diag(Z[np.diag_indices_from(Z)]))[triu].astype(np.double)
    # N is the row-sums of Z
    N = np.sum(Z, axis=1).astype(np.double)
    # K are the indepdent variables we're sampling, which conceptually represent
    # a symmetric non-negative 2D "virtual" count matrix. We're storing them
    # in symmetric storage.
    Z_sum = np.sum(Z, dtype=float)
    K = (0.5 * (Z + Z.T) / Z_sum)[triu]
    # Q is the row-sums of 2D matrix, K. Note that since K is in symmetric
    # storage, we need to be a little careful.
    Q = 0.5 * (np.sum(Z, axis=1) + np.sum(Z, axis=0)) / Z_sum

    
    print "S", S
    print 'N', N
    print 'K', K
    print 'Q', Q
    print '\n\n'
    
    
    
    

def metzner_mcmc_slow(Z, n_samples, n_thin=1, random_state=None):
    """Metropolis Markov chain Monte Carlo sampler for reversible transition
    matrices

    Parameters
    ----------
    Z : np.array, shape=(n_states, n_states)
    n_samples : int
    n_thin : int
    random_state : {None, int, RandomState}

    Returns
    -------

    References
    ----------
    .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
        Distribution of transition matrices and functions of transition
        matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)

    Notes
    -----
    This implementation is not optimized.
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
        i, j = random.randint(n_states, size=2)
        i, j = (0, 2)
        
        sc = np.sum(K)
        if i == j:
            a, b = max(-K[i,j], K_MINUS - sc), K_PLUS - sc
        else:
            a, b = max(-K[i,j], 0.5*(K_MINUS - sc)), 0.5*(K_PLUS - sc)

        epsilon = random.uniform(a, b)
        epsilon = 0.01
        K_proposal = np.copy(K)
        K_proposal[i, j] += epsilon
        if i != j:
            K_proposal[j, i] += epsilon

        # acceptance?
        cutoff = np.exp(_logprob_T(_K_to_T(K_proposal), Z) -
                        _logprob_T(_K_to_T(K), Z))
        r = random.rand()
        print "i=%d, j=%d" % (i, j)
        print "a=%f, b=%f" % (a, b)
        print "epsilon=%f" % epsilon
        print "cutoff=%f" % cutoff
        # print "new logprob", _logprob_T(_K_to_T(K_proposal), Z)
        print "old logprob", _logprob_T(_K_to_T(K), Z)
        print "old logprob 1", np.sum(np.multiply(Z, np.log(K)))
        print "old logprpob 2", -np.sum(np.multiply(np.sum(Z, axis=1), np.log(np.sum(K, axis=1))))
        print "sum of two old logprob",  np.sum(np.multiply(Z, np.log(K))) - np.sum(np.multiply(np.sum(Z, axis=1), np.log(np.sum(K, axis=1))))
        print "Q", np.sum(K, axis=1)
        print "N", np.sum(Z, axis=1)
        print Z.flatten(), np.log(K).flatten()
        print K

        if r < cutoff:
            n_accept += 1
            K = K_proposal

        if t % n_thin == 0:
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


def main():
    C = np.array([[1, 10, 2], [2, 26, 3], [15, 20, 20]])
    list(metzner_mcmc_slow(C, 1, random_state=0))
    print metzner_mcmc_fast(C, 1)

if __name__ == '__main__':
    # _metzner_figure_4()
    main()
