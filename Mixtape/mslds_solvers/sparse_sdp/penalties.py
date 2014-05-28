import numpy as np
import scipy.misc
"""
Various Useful penalty Functions.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com

TODO:
    -) Factor common penalty calculation out into separate function
"""

def compute_scale(m, n, p, q, eps):
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
    M = 1
    if m + n + p + q> 0:
        M = max((np.log(m+1) + np.log(n+1)
                + np.log(p+1) + np.log(q+1)), 1.) / eps
    return M

def penalties(X, As, bs, Cs, ds, Fs, Gs):
    """
    Computes penalties

     (max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|},
      max_k {Fk(x)}, max_l {|Gl(x)|})

    """
    (dim, _) = np.shape(X)
    m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
    penalties = np.zeros(n+m+p+q)
    # Handle linear inequalities
    for i in range(m):
        penalties[i] = np.trace(np.dot(As[i],X)) - bs[i]
    # Handle linear equalities
    for j in range(n):
        penalties[j+m] = np.abs(np.trace(np.dot(Cs[j],X)) - ds[j])
    # Handle convex inequalities
    for k in range(p):
        penalties[k+m+n] = Fs[k](X)
    # Handle convex equalities
    for l in range(q):
        penalties[l+p+m+n] = np.abs(Gs[l](X))
    return penalties

def log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs):
    """
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
    pens = penalties(X, As, bs, Cs, ds, Fs, Gs)
    retval = 0.
    m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
    if m + n + p + q > 0:
        try:
            retval = scipy.misc.logsumexp(M*np.array(pens), axis=0)
            retval = -(1.0/M) * retval
        except FloatingPointError:
            if np.amax(pens) == np.inf:
                return -np.inf
    return retval

def log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
    """
   Computes grad f(X) = -(1/M) * c' / c where
   c' = (sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M*Ai.T)
       + sum_{j=1}^n exp(M*|Tr(Cj,X) - dj|) * sgn(Tr(Cj,X) - dj) * (M*Cj.T)
       + sum_{k=1}^p exp(M*f_k(X)) * (M*f_k'(X))
       + sum_{l=1}^q exp(M*|g_l(X)|) * sgn(g_l(X)) * (M*g_l'(X)))
   c  = (sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
       + sum_{i=1}^n exp(M*|Tr(Cj,X) - dj|)
       + sum_{k=1}^p exp(M*f_k(X))
       + sum_{l=1}^q exp(M*|g_l(X)|))

    As[i] and Cs[j] should be symmetric real matrices and Fs[k], Gs[l] to
    be convex functions.
    """
    (dim, _) = np.shape(X)
    m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
    retval = 0.
    num_mats = []
    if m+n+p+q <= 0:
        return None
    log_nums = np.zeros(m+n+p+q)
    for i in range(m):
        log_nums[i] = M*(np.trace(np.dot(As[i],X)) - bs[i])
        num_mats.append(M*As[i].T)
    for j in range(n):
        log_nums[j+m] = M*np.abs(np.trace(np.dot(Cs[j],X)) - ds[j])
        val = np.trace(np.dot(Cs[j],X)) - ds[j]
        if val < 0 :
            sgn = -1.
        else:
            sgn = 1.
        num_mats.append(sgn*M*Cs[j].T)
    for k in range(p):
        log_nums[k+n+m] = M*Fs[k](X)
        num_mats.append(M*gradFs[k](X))
    for l in range(q):
        log_nums[l+n+m+p] = M*np.abs(Gs[l](X))
        val = Gs[l](X)
        if val < 0 :
            sgn = -1.
        else:
            sgn = 1.
        num_mats.append(sgn*M*gradGs[l](X))
    log_nums = np.array(log_nums)
    log_denom = scipy.misc.logsumexp(log_nums, axis=0)

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

def neg_max_penalty(X, As, bs, Cs, ds, Fs, Gs):
    return -np.amax(penalties(X, As, bs, Cs, ds, Fs, Gs))

def neg_max_grad_penalty(X, As, bs, Cs, ds, Fs, gradFs, Gs,
        gradGs, eps):
    """
    Gradient of neg_max_penalty
    Parameters
    __________

    X: numpy.ndarray
        input variable
    As: list
        constraints Tr(A_iX) <= b_i
    bs: list
        constraints Tr(A_iX) <= b_i
    Cs: list
        constraints Tr(C_iX) == d_i
    ds: list
        constraints Tr(C_iX) == d_i
    Fs: list
        constraints f_i(X) <= 0
    Gs: list
        constraints g_i(X) == 0
    dim: int
        Input dimension. X is a (dim, dim) array.
    eps: float
        error tolerance.
    """
    # should assert X is a square matrix...
    (dim, _) = np.shape(X)
    m = len(As)
    n = len(Cs)
    p = len(Fs)
    q = len(Gs)
    pens = penalties(X, As, bs, Cs, ds, Fs, Gs)
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
