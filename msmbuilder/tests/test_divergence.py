import numpy as np
from msmbuilder.utils.divergence import _kld
from scipy.stats import entropy

def _get_random_prob_dist(n):
    P = np.random.random([n,n])
    Q = np.random.random([n,n])
    P = (P.T / np.sum(P,axis=1)).T
    Q = (Q.T / np.sum(Q,axis=1)).T
    return P, Q

def test_kullback_leibler_manual():
    P, Q = _get_random_prob_dist(4)

    vec = []
    for row in range(P.shape[0]):
        temp = 0
        for i, entry in enumerate(P[row]):
            temp += entry * np.log(entry/Q[row][i])
        vec.append(temp)
    manual_kl = np.array(vec)

    msmb_kl = _kld(P, Q, scalar=False)

    assert np.allclose(manual_kl, msmb_kl)


def test_kullback_leibler_scipy():
    P, Q = _get_random_prob_dist(4)

    scipy_kl = entropy(P.T, Q.T)
    msmb_kl = _kld(P, Q, scalar=False)  

    assert np.allclose(scipy_kl, msmb_kl)
