import numpy as np
from msmbuilder.utils import divergence
from scipy.stats import entropy

def test_kullback_leibler():
    P = np.random.random([4,4])
    Q = np.random.random([4,4])
    P = (P.T / np.sum(P,axis=1)).T
    Q = (Q.T / np.sum(Q,axis=1)).T

    vec = []
    for row in range(P.shape[0]):
        temp = 0
        for i, entry in enumerate(P[row]):
            temp += entry * np.log(entry/Q[row][i])
        vec.append(temp)
    manual_kl = np.array(vec)

    scipy_kl = entropy(P.T, Q.T)

    print(manual_kl)
    print(scipy_kl)

    assert np.allclose(manual_kl, scipy_kl)
