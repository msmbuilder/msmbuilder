cpdef ldirichlet_softmax(const double[::1] theta, const double[::1] alpha,
                             double[::1] grad=None):
    r"""Unnormalized log probability density of Dirichlet distribution
    in the 'softmax' reparameterization.

    This includes a Gaussian prior on the unconstrained direction. See Eq. 16
    from [1], with \epsilon = 1.

    Parameters
    ----------
    alpha : ndarray, shape=(n_params,)
        Dirichlet concentration parameter.
    theta : ndarray, shape=(n_params,)
        Unnormalized log probabilities. The actually Dirichlet distributed
        variables are ``exp(theta) / sum(exp(theta))``.
    grad : ndarray, shape=(n_params,)
        If not None, on exit, grad will be incremented by the gradient
        of ``logprob`` with respect to ``theta``.

    Returns
    -------
    logprob : float
        Unnormalized log probability of ``theta``.

    References
    ----------
    .. [1] MacKay, D. J. C. "Choice of basis for Laplace approximation."
       Machine learning 33.1 (1998): 77-86.
    """
    cdef npy_intp i, n
    cdef double[::1] exptheta, pi
    cdef double logp = 0
    cdef double norm = 0, sumtheta = 0, sumalpha = 0, lognorm = 0
    n = theta.shape[0]
    if theta.shape[0] != alpha.shape[0]:
        raise ValueError('len(theta) != len(alpha)')
    if grad is not None and theta.shape[0] != grad.shape[0]:
        raise ValueError('len(theta) != len(grad)')

    logpi = zeros(n)
    exptheta = zeros(n)

    for i in range(n):
        exptheta[i] = exp(theta[i])
        norm += exptheta[i]
        sumtheta += theta[i]
        sumalpha += alpha[i]

    lognorm = log(norm)
    for i in range(n):
        logpi[i] = theta[i] - lognorm

    cddot(alpha, logpi, &logp)
    logp -= sumtheta*sumtheta / 2

    # logp = np.dot(alpha, np.log(pi)) - sum_theta**2 / 2
    # grad = -exptheta * (sumalpha / norm) + alpha - sumtheta
    if grad is not None:
        for i in range(n):
            grad[i] += (-exptheta[i] * (sumalpha / norm) + alpha[i] - sumtheta)

    return logp


def lexponential(const double[::1] theta, const double[::1] beta,
                 double[::1] grad=None):
    r"""Log probability of the exponential distribution ::

        f(\theta; \beta) = \frac{1}{\beta} \exp(-\frac{\theta}{\beta})

    Parameters
    ----------
    theta : ndarray, shape=(n_params,)
        The free parameters.
    beta : ndarray, shape=(n_params,)
        The scale parameters of the distribution. Beta must be greater
        than zero.
    grad : ndarray, shape=(n_params,)
        If not None, on exit, grad will be incremented by the gradient
        of ``logprob`` with respect to ``theta``.

    Returns
    -------
    logprob : float
        Log probability of ``theta``.
    """
    cdef npy_intp i, n
    cdef double[::1] exptheta
    cdef double logp = 0
    cdef double norm = 0
    n = theta.shape[0]
    if theta.shape[0] != beta.shape[0]:
        raise ValueError('len(theta) != len(beta)')
    if grad is not None and theta.shape[0] != grad.shape[0]:
        raise ValueError('len(theta) != len(grad)')

    for i in range(n):
        logp -= log(beta[i]) + theta[i] / beta[i]

    if grad is not None:
        for i in range(n):
            grad[i] -= 1 / beta[i]

    return logp
