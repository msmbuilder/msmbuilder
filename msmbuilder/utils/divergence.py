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


def _jsd(P, Q, scalar=True):
    M = np.mean([P, Q], axis=0)
    return 0.5 * _kld(P, M, scalar=scalar) + 0.5 * _kld(Q, M, scalar=scalar)


def kl_divergence(target, ref, i):
    return np.array([_kld(ref[i],t) for t in target])


def symmetric_kl_divergence(target, ref, i):
    return np.array([_sym_kld(ref[i],t) for t in target])


def js_divergence(target, ref, i):
    return np.array([_jsd(ref[i],t) for t in target])


def _make_square(sequence):
    n_states = int(np.sqrt(len(sequence)))
    return np.array([x.reshape(n_states, n_states) for x in sequence])


def kl_divergence_msm(target, ref, i):
    return kl_divergence(_make_square(target), _make_square(ref), i)


def symmetric_kl_divergence_msm(target, ref, i):
    return symmetric_kl_divergence(_make_square(target), _make_square(ref), i)


def js_divergence_msm(target, ref, i):
    return js_divergence(_make_square(target), _make_square(ref), i)
