import numpy as np
from sklearn.utils import check_random_state


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
        cutoff = min(1, np.exp(_logprob_T(_K_to_T(K_proposal), Z) - 
                               _logprob_T(_K_to_T(K), Z)))
        r = random.rand()
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

if __name__ == '__main__':
    _metzner_figure_4()
