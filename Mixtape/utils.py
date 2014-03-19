import json
import numpy as np
from numpy import shape
from sklearn.utils import check_random_state
from numpy.linalg import norm
from numpy.random import randint
import mdtraj as md


def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass


def categorical(pvals, size=None, random_state=None):
    """Return random integer from a categorical distribution

    Parameters
    ----------
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1.
    size : int or tuple of ints, optional
        Defines the shape of the returned array of random integers. If None
        (the default), returns a single float.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    """
    cumsum = np.cumsum(pvals)
    if size is None:
        size = (1,)
        axis = 0
    elif isinstance(size, tuple):
        size = size + (1,)
        axis = len(size) - 1
    else:
        raise TypeError('size must be an int or tuple of ints')

    random_state = check_random_state(random_state)
    return np.sum(cumsum < random_state.random_sample(size), axis=axis)


##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V


def assignment_to_weights(assignments, K):
    """Turns a hard assignment into a weights matrix. Useful for
       experimenting with Viterbi-learning.
    """
    (T,) = np.shape(assignments)
    W_i_Ts = np.zeros((T, K))
    for t in range(T):
        ind = assignments[t]
        for k in range(K):
            if k != ind:
                W_i_Ts[t, k] = 0.0
            else:
                W_i_Ts[t.__int__(), ind.__int__()] = 1.0
    return W_i_Ts


def empirical_wells(Ys, W_i_Ts):
    (T, y_dim) = np.shape(Ys)
    (_, K) = np.shape(W_i_Ts)
    means = np.zeros((K, y_dim))
    covars = np.zeros((K, y_dim, y_dim))
    for k in range(K):
        num = np.zeros(y_dim)
        denom = 0
        for t in range(T):
            num += W_i_Ts[t, k] * Ys[t]
            denom += W_i_Ts[t, k]
        means[k] = (1.0 / denom) * num
    for k in range(K):
        num = np.zeros((y_dim, y_dim))
        denom = 0
        for t in range(T):
            num += W_i_Ts[t, k] * np.outer(Ys[t] - means[k],
                                           Ys[t] - means[k])
            denom += W_i_Ts[t, k]
        covars[k] = (1.0 / denom) * num
    return means, covars


def means_match(base_means, means, assignments):
    (K, y_dim) = np.shape(means)
    (T,) = np.shape(assignments)
    matching = np.zeros(K)
    new_assignments = np.zeros(T)
    for i in range(K):
        closest = -1
        closest_dist = Inf
        for j in range(K):
            if norm(base_means[i] - means[j]) < closest_dist:
                closest = j
                closest_dist = norm(base_means[i] - means[j])
        matching[i] = closest
    for t in range(T):
        new_assignments[t] = matching[assignments[t]]
    return matching, new_assignments
