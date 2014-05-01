import numpy as np
import scipy
"""
Various Useful Penalty Functions for Hazan's Algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com

TODOs: Too many of the penalties below are similar. Here are some
simplifying steps.

    -) Remove old neg_max/log_sum_exp penalties and gradients and rename
       neg_max_general penalties and gradients to neg_max.
    -) Factor out penalty calculation into shared subfunction
       to avoid duplication in both penalty and gradient functions.
    -) Change penalty functions to no longer require m, n, dim, etc.
"""

def compute_scale(m, n, eps):
    return compute_scale_full(m, n, 0, 0, eps)

def compute_scale_full(m, n, p, q, eps):
    """
    Compute the scaling factor required for m inequality and
    n equality constraints in the log_sum_exp penalty.

    Parameters
    __________
    m: int
        Number of affine inequality constraints
    n: int
        Number of affine equality constraints
    p: int
        Number of convex inequality constraints
    q: int
        Number of convex equality constraints
    """
    if m + n + p + q> 0:
        M = np.max((np.log(m+1) + np.log(n+1)
                    + np.log(p+1) + np.log(q+1)), 1.) / eps

        if m > 0:
            M += np.max((np.log(m), 1.))/eps
        if n > 0:
            M += np.max((np.log(n), 1.))/eps
        if p > 0:
            M += np.max((np.log(p), 1.))/eps
        if q > 0:
            M += np.max((np.log(q), 1.))/eps
    else:
        M = 1.
    return M


def neg_sum_squares(x):
    """
    Computes f(x) = -\sum_k x_kk^2. Useful for debugging.

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    retval = 0.
    for i in range(N):
        retval += -x[i,i]**2
    return retval

def grad_neg_sum_squares(x):
    """
    Computes grad(-\sum_k x_kk^2). Useful for debugging.

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    G = np.zeros((N,N))
    for i in range(N):
        G[i,i] += -2.*x[i,i]
    return G

def log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs):
    """
    TODO: Make this more numerically stable
    Computes
    f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
                    + sum_{j=1}^n exp(M*|Tr(Cj,X) - dj|)
                    + sum_{k=1}^p exp(M*f_k(X))
                    + sum_{l=1}^q exp(M*|g_l(X)|))

    where m is the number of linear constraints and M = log m / eps,
    with eps an error tolerance parameter

    Parameters
    __________

    X: numpy.ndarray
       Input matrix
    M: float
        Rescaling Factor
    As: list
        Inequality matrices
    bs: list
        Inequality vectors
    Cs: list
        Equality matrices
    ds: list
        Equality vectors
    Fs: list
        Convex function inequalities
    Gs: list
        Convex function equalities
    """
    (dim, _) = np.shape(X)
    m = len(As)
    n = len(Cs)
    p = len(Fs)
    q = len(Gs)
    penalties = np.zeros(m+n+p+q)
    # Handle linear inequalities
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        penalties[i] = M*(np.trace(np.dot(Ai,X)) - bi)
    # Handle linear equalities
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        penalties[j+m] = M*np.abs(np.trace(np.dot(Cj,X)) - dj)
    # Handle convex inequalities
    for k in range(p):
        Fk = Fs[k]
        penalties[k+m+n] = Fk(X)
    # Handle convex equalities
    for l in range(q):
        Gl = Gs[l]
        penalties[l+p+m+n] = np.abs(Gl(X))
    retval = 0.
    if m + n + p + q > 0:
        retval = scipy.misc.logsumexp(np.array(penalties), axis=0)
        retval = -(1.0/M) * np.exp(retval)
    return retval

def log_sum_exp_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps):
    """
    Computes grad f(X) = -(1/M) * c' / c where
      c' = (sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
            + sum_{j=1}^n exp(M*|Tr(Cj,X) - dj|)
                            * sgn(Tr(Cj,X) - dj) * Cj.T)
      c  = (sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            + sum_{i=1}^n exp(M*|Tr(Cj,X) - dj|))

    Need Ai and Cj to be symmetric real matrices. The exponent terms
    can become large and cause overflow issues. As a result,
    we need to log-sum-exp the exponent terms to aboid this problem.
    """
    retval = 0.
    num_mats = []
    log_nums = np.zeros(m+n+p+q)
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        num_mats.append(M*Ai.T)
        log_nums[i] = M*(np.trace(np.dot(Ai,X)) - bi)
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        val = np.trace(np.dot(Cj,X)) - dj
        if val < 0 :
            sgn = -1.
        else:
            sgn = 1.
        num_mats.append(sgn*Cj.T)
        log_nums[j+m] = M*np.abs(np.trace(np.dot(Cj,X)) - dj)
    for k in range(p):
        pass
    for l in range(q):
        pass
    if m + n + p + q > 0:
        log_denom = scipy.misc.logsumexp(np.array(log_nums), axis=0)

        # Now subtract from numerator
        log_nums -= log_denom

        # Undo logarithmic storage
        nums = np.exp(log_nums)

        # Now construct gradient
        grad = np.zeros(np.shape(X))
        for ind in range(m+n+p+q):
            grad += nums[ind] * num_mats[ind]
        grad = -(1.0/M) * grad
    return grad

def neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs):
    penalties = neg_max_penalties(X, M, As, bs, Cs, ds, Fs, Gs)
    return -np.amax(penalties)

def neg_max_penalties(X, M, As, bs, Cs, ds, Fs, Gs):
    """
    Computes penalty

     -max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|},
            max_k {Fk(x)}, max_l {|Gl(x)|})

    """
    (dim, _) = np.shape(X)
    m = len(As)
    n = len(Cs)
    p = len(Fs)
    q = len(Gs)
    penalties = np.zeros(n+m+p+q)
    # Handle linear inequalities
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            penalties[i] = np.trace(np.dot(Ai,X)) - bi
        else:
            penalties[i] = Ai*X - bi
    # Handle linear equalities
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            penalties[j+m] = np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            penalties[j+m] = np.abs(Cj*X - dj)
    # Handle convex inequalities
    for k in range(p):
        Fk = Fs[k]
        penalties[k+m+n] = Fk(X)
    # Handle convex equalities
    for l in range(q):
        Gl = Gs[l]
        penalties[l+p+m+n] = np.abs(Gl(X))
    return penalties

def neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs, Gs,
        gradGs, eps):
    """
    A more complicated version of neg_max_grad_penalty that allows for
    arbitrary convex inequalities and equalities.
    Parameters
    __________

    X: numpy.ndarray
        input variable
    As: list
        list of numpy.ndarrays for constraints Tr(A_iX) <= b_i
    bs: list
        list of floats for constraints Tr(A_iX) <= b_i
    Cs: list
        list of numpy.ndarrays for constraints Tr(C_iX) == d_i
    ds: list
        list of floats for constraints Tr(C_iX) == d_i
    Fs: list
        list of functions for constraints f_i(X) <= 0
    Gs: list
        list of functions for constraints g_i(X) == 0
    dim: int
        Input dimension. X is a (dim, dim) array.
    eps: float
        eps > 0 is the error tolerance.
    """
    # should assert X is a square matrix...
    (dim, _) = np.shape(X)
    m = len(As)
    n = len(Cs)
    p = len(Fs)
    q = len(Gs)
    penalties = neg_max_penalties(X, M, As, bs, Cs, ds, Fs, Gs)
    inds = [ind for ind in range(n+m+p+q) if penalties[ind] > eps]
    grad = np.zeros(np.shape(X))
    pen_sum = 0
    for ind in inds:
        pen_sum += penalties[ind]
        if ind < m:
            Ai = As[ind]
            grad += penalties[ind] * Ai
        elif ind < m+n:
            Cj = Cs[ind - m]
            dj = ds[ind - m]
            val = np.trace(np.dot(Cj,X)) - dj
            if val < 0:
                grad += - penalties[ind] * Cj
            elif val > 0:
                grad += penalties[ind] * Cj
        elif ind < m+n+p:
            gradFj = gradFs[ind]
            grad += penalties[ind] * gradFj(X)
        else:
            Gl = Gs[ind]
            gradGl = gradGs[ind]
            val = Gl(X)
            if val < 0:
                grad += -penalties[ind] * gradGl(X)
            else:
                grad += penalties[ind] * gradGl(X)
    pen_sum = max(pen_sum, 1.0)
    # Average by penalty sum
    grad = grad / pen_sum
    # Average by num entries
    grad = grad / max(len(inds), 1.)
    # Take the negative since our function is -max{..}
    grad = -grad
    return grad
