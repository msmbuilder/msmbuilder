import numpy as np
import scipy
"""
Various Useful Penalty Functions for Hazan's Algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""

def compute_scale(m,n, eps):
    """
    Compute the scaling factor required for m inequality and
    n equality constraints in the log_sum_exp penalty.

    Parameters
    __________
    m: int
        Number of inequality constraints
    n: int
        Number of equality constraints
    """
    if m + n > 0:
        M = 0.
        if m > 0:
            M += np.max((np.log(m), 1.))/eps
        if n > 0:
            M += np.max((np.log(n), 1.))/eps
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

def log_sum_exp_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    TODO: Make this more numerically stable
    Computes
    f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
                    + sum_{j=1}^n exp(M*|Tr(Cj,X) - dj|))

    where m is the number of linear constraints and M = log m / eps,
    with eps an error tolerance parameter

    Parameters
    __________

    m: int
        Number of inequaltity constraints
    n: int
        Number of equality constraints
    M: float
        Rescaling Factor
    """
    s = None
    r = None
    penalties_m = np.zeros(m)
    penalties_n = np.zeros(n)
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            penalties_m[i] = M*(np.trace(np.dot(Ai,X)) - bi)
        else:
            penalties_m[i] = M*(Ai*X - bi)
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            penalties_n[j] = M*np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            penalties_n[j] = M*np.abs(Cj*X - dj)
    retval = 0.
    if m + n > 0:
        if m > 0:
            retval += scipy.misc.logsumexp(np.array(penalties_m), axis=0)
        if n > 0:
            retval += scipy.misc.logsumexp(np.array(penalties_n), axis=0)
        retval = -(1.0/M) * np.exp(retval)
    return retval

def log_sum_exp_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps):
    """
    TODO: Make this more numerically stable
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
    num_mats_m = []
    log_nums_m = np.zeros(m)
    log_denoms_m = np.zeros(m)
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            num_mats_m.append(M*Ai.T)
            log_nums_m[i] = M*(np.trace(np.dot(Ai,X)) - bi)
            log_denoms_m[i] = M*(np.trace(np.dot(Ai,X)) - bi)
        else:
            num_mats_m.append(M*Ai.T)
            log_nums_m[i] = M*(Ai*X - bi) + (M*Ai.T)
            log_denoms_m[i] = M*(Ai*X - bi)
    num_mats_n = []
    log_nums_n = np.zeros(n)
    log_denoms_n = np.zeros(n)
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            val = np.trace(np.dot(Cj,X)) - dj
            if val < 0 :
                sgn = -1.
            else:
                sgn = 1.
            # fix derivative
            num_mats_n.append(sgn*Cj.T)
            log_nums_n[j] = M*np.abs(np.trace(np.dot(Cj,X)) - dj)
            log_denoms_n[j] = M*np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            val = (Cj*X) - dj
            if val < 0 :
                sgn = -1.
            else:
                sgn = 1.
            num_mats_n.append(sgn*Cj.T)
            log_nums_n[j] = M*np.abs(Cj*X - dj)
            log_denoms_n[j] = M*np.abs(Cj*X - dj)
    if m + n > 0:
        log_denom_m = -np.inf
        log_denom_n = -np.inf
        if m > 0:
            log_denom_m = scipy.misc.logsumexp(np.array(log_denoms_m),
                                        axis=0)
        if n > 0:
            log_denom_n = scipy.misc.logsumexp(np.array(log_denoms_n),
                                        axis=0)
        log_denom = np.logaddexp(log_denom_m, log_denom_n)

        # Now subtract from numerator
        log_nums_m -= log_denom
        log_nums_n -= log_denom

        # Undo logarithmic storage
        nums_m = np.exp(log_nums_m)
        nums_n = np.exp(log_nums_n)

        # Now construct gradient
        grad = np.zeros(np.shape(X))
        #grad += np.sum(nums_m * num_mats_m)
        #grad += np.sum(nums_n * num_mats_n)
        for i in range(m):
            grad += nums_m[i] * num_mats_m[i]
        for j in range(n):
            grad += nums_n[j] * num_mats_n[j]
        grad = -(1.0/M) * grad
    #import pdb
    #pdb.set_trace()
    return grad

def neg_max_general_penalty(X, M, As, bs, Cs, ds, Fs, Gs):
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
    return -np.amax(penalties)

def neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    Computes penalty

     -max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|})

    """
    penalties = np.zeros(n+m)
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            penalties[i] = np.trace(np.dot(Ai,X)) - bi
        else:
            penalties[i] = Ai*X - bi
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            penalties[j+m] += np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            penalties[j+m] += np.abs(Cj*x - dj)
    return -np.amax(penalties)

def neg_max_general_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs, Gs,
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
    inds = [ind for ind in range(n+m) if penalties[ind] > eps]
    grad = np.zeros(np.shape(X))
    for ind in inds:
        if ind < m:
            Ai = As[ind]
            grad += Ai
        elif ind < m+n:
            Cj = Cs[ind - m]
            dj = ds[ind - m]
            val = np.trace(np.dot(Cj,X)) - dj
            #print "val: ", val
            if val < 0:
                grad += -Cj
            elif val > 0:
                grad += Cj
        elif ind < m+n+p:
            gradFj = gradFs[ind]
            grad += gradFj(X)
        else:
            Gl = Gs[ind]
            gradGl = gradGs[ind]
            val = Gl(X)
            if val < 0:
                grad += -gradGl(X)
            else:
                grad += gradGl(X)
    # Average by num entries
    grad = grad / max(len(inds), 1.)
    # Take the negative since our function is -max{..}
    grad = -grad
    #import pdb
    #pdb.set_trace()
    return grad

def neg_max_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps):
    """
    Note that neg_max_penalty(X) roughly equals

    -max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|})

    Since neg_max_penalty is concave, we need to return a supergradient
    of this function. The supergradient of this function equals the
    subgradient of -neg_max_penalty(X):

    max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|})

    which equals

    Conv{{A_i | -neg_max_penalty(X) = Tr(Ai,X) - bi} union
         { sign(Tr(Cj,X) - dj) Cj  | penalty(X) = |Tr(Cj,X) - dj| }}

    We use a weak subdifferential calculus that averages the gradients
    of all violated constraints.

    Parameters
    __________

    X: numpy.ndarray
        input variable
    m: int
        number of linear inequality constraints
    n: int
        number of linear equality constraints
    As: list
        list of numpy.ndarrays for constraints Tr(A_iX) <= b_i
    bs: list
        list of floats for constraints Tr(A_iX) <= b_i
    Cs: list
        list of numpy.ndarrays for constraints Tr(C_iX) == d_i
    ds: list
        list of floats for constraints Tr(C_iX) == d_i
    dim: int
        Input dimension. X is a (dim, dim) array.
    eps: float
        eps > 0 is the error tolerance.
    """
    penalties = np.zeros(n+m)
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
    inds = [ind for ind in range(n+m) if penalties[ind] > eps]
    grad = np.zeros(np.shape(X))
    for ind in inds:
        if ind < m:
            Ai = As[ind]
            grad += Ai
        else:
            Cj = Cs[ind - m]
            dj = ds[ind - m]
            val = np.trace(np.dot(Cj,X)) - dj
            #print "val: ", val
            if val < 0:
                grad += -Cj
            elif val > 0:
                grad += Cj
    # Average by num entries
    grad = grad / max(len(inds), 1.)
    # Take the negative since our function is -max{..}
    grad = -grad
    #import pdb
    #pdb.set_trace()
    return grad
