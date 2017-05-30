from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.stats import entropy

def scipy_kl_divergence(P, Q, scalar=True):
    result = entropy(P.T, Q.T)
    if scalar:
        return np.sum(result)
    else:
        return result


def manual_kl_divergence(P, Q, scalar=True):
    if len(P.shape) == 1:
        P = np.array([P])
        Q = np.array([Q])
    vec = []
    for row in range(P.shape[0]):
        temp = 0
        for i, entry in enumerate(P[row]):
            if entry*Q[row][i] != 0: # i.e. one or both is not zero
                temp += entry * np.log(entry/Q[row][i])
        vec.append(temp)
    result = np.array(vec)
    if scalar:
        return np.sum(result)
    else:
        return result


def kl_divergence(P, Q, manual=True, scalar=True):
    if manual:
        return manual_kl_divergence(P, Q, scalar=scalar)
    else:
        return scipy_kl_divergence(P, Q, scalar=scalar)


def sym_kl_divergence(P, Q, scalar=True):
    return kl_divergence(P,Q,scalar=scalar) + kl_divergence(Q,P,scalar=scalar)


def js_divergence(P, Q, scalar=True):
    M = np.mean([P, Q], axis=0)
    return (0.5 * kl_divergence(P, M, scalar=scalar) +
            0.5 * kl_divergence(Q, M, scalar=scalar))


def js_metric(P, Q, scalar=True):
    return np.sqrt(js_divergence(P, Q, scalar=scalar))


def fnorm(P, Q):
    return np.linalg.norm(P - Q, ord='fro')


def kl_divergence_array(target, ref, i):
    return np.array([kl_divergence(ref[i],t) for t in target])


def sym_kl_divergence_array(target, ref, i):
    return np.array([sym_kl_divergence(ref[i],t) for t in target])


def js_divergence_array(target, ref, i):
    return np.array([js_divergence(ref[i],t) for t in target])


def js_metric_array(target, ref, i):
    return np.array([js_metric(ref[i],t) for t in target])


def fnorm_array(target,ref, i):
    return np.array([fnorm(ref[i],t) for t in target])


def _make_square(sequence):
    n_states = int(np.sqrt(len(sequence[0])))
    return np.array([x.reshape(n_states, n_states) for x in sequence])


def kl_divergence_msm(target, ref, i):
    return kl_divergence_array(_make_square(target), _make_square(ref), i)


def sym_kl_divergence_msm(target, ref, i):
    return sym_kl_divergence_array(_make_square(target), _make_square(ref), i)


def js_divergence_msm(target, ref, i):
    return js_divergence_array(_make_square(target), _make_square(ref), i)


def js_metric_msm(target, ref, i):
    return js_metric_array(_make_square(target), _make_square(ref), i)


def fnorm_msm(target, ref, i):
    return fnorm_array(_make_square(target), _make_square(ref), i)

