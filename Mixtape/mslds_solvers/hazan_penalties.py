import numpy as np
"""
Various Useful Penalty Functions for Hazan's Algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""

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
                    + sum_{j=1}^n exp(M*(Tr(Cj,X) - dj)^2))

    Parameters
    __________

    m: int
        Number of inequaltity constraints
    n: int
        Number of equality constraints
    M: float
        Rescaling Factor
    """
    s = 0.
    r = 0.
    retval = 0.
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            s += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
        else:
            s += np.exp(M*(Ai*X - bi))
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            r += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
        else:
            r += np.exp(M*(Cj*X - dj)**2)
    if m + n > 0:
        retval += -(1.0/M) * np.log(s + r)
    return retval

def log_sum_exp_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    TODO: Make this more numerically stable
    Computes grad f(X) = -(1/M) * c' / c where
      c' = (sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
            + sum_{j=1}^n exp(M(Tr(Cj,X) - dj)**2)
                            * (2M(Tr(Cj,X) - dj)) * Cj.T)
      c  = (sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            + sum_{i=1}^n exp(M(Tr(Cj,X) - dj)**2))

    Need Ai and Cj to be symmetric real matrices
    """
    retval = 0.
    num = 0.
    denom = 0.
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            num += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))*(M*Ai.T)
            denom += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
        else:
            num += np.exp(M*(Ai*X - bi))*(M*Ai.T)
            denom += np.exp(M*(Ai*X - bi))
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            num += (np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)*
                    (2*M*(np.trace(np.dot(Cj,X)) - dj))*
                    Cj.T)
            denom += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
        else:
            num += np.exp(M*(Cj*x - dj)**2)*(2*M*(Cj*x - dj))*Cj.T
            denom += np.exp(M*(Cj*x - dj)**2)
    if m + n > 0:
        retval += -(1.0/M) * num/denom
    #import pdb
    #pdb.set_trace()
    return retval


def neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    Note that penalty(X) roughly equals

     -max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|})

    This function computes and returns this quantity.
    """
    penalties = np.zeros(n+m)
    count = 0
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

def neg_max_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps):
    """
    Note that penalty(X) roughly equals

     max(max_i {Tr(Ai,X) - bi}, max_j{(Tr(Cj,X) - dj)^2})

    The subgradient of this quantity is given by

    Conv{{A_i | penalty(X) = Tr(Ai,X) - bi} union
         { sign(Tr(Cj,X) - dj) 2 Cj  | penalty(X) = |Tr(Cj,X) - dj| }}

    We use a weak subdifferential calculus that averages the gradients
    of all violated constraints.
    """
    penalties = np.zeros(n+m)
    count = 0
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
    #ind = np.argmax(penalties)
    inds = [ind for ind in range(n+m) if penalties[ind] > eps]

    grad = np.zeros(np.shape(X))
    for ind in inds:
        if ind < m:
            Ai =  As[ind]
            grad += Ai
        else:
            Cj = Cs[ind - m]
            val = np.trace(np.dot(Cj,X)) - dj
            if val < 0:
                grad += - Cj
            elif val > 0:
                grad += Cj
    # Average by num entries
    grad = grad / max(len(inds), 1.)
    # Take the negative since our function is -max{..}
    return -grad

