from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.stats import entropy

def _kld(P, Q, scalar=True):
    result = entropy(P.T, Q.T)
    if scalar:
        return np.sum(result)
    else:
        return result


def _sym_kld(P, Q, scalar=True):
    return _kld(P,Q,scalar=scalar) + _kld(Q,P,scalar=scalar)


def _get_mean(P, Q):
    if P.shape != Q.shape:
        raise ValueError('P and Q must have the same shape')
    M = []
    for row in range(P.shape[0]):
        temp = []
        for i, entry in enumerate(P[row]):
            temp.append((entry + Q[row][i])/2.)
        M.append(temp)
    return np.array(M)


def _jsd(P, Q, scalar=True):
    M = _get_mean(P, Q)
    return 0.5 * _kld(P, M, scalar=scalar) + 0.5 * _kld(Q, M, scalar=scalar)


def kl_divergence(target, ref, i):
    return np.array([_kld(ref[i],t) for t in target])


def symmetric_kl_divergence(target, ref, i):
    return np.array([_sym_kld(ref[i],t) for t in target])


def js_divergence(target, ref, i):
    return np.array([_jsd(ref[i],t) for t in target])
