import numpy as np
import scipy.linalg


def contraction(transmat, lag_time, pi=None):
    """Contract a row-stochastic transition matrix from a time-discretization
    `lag_time` to a time-discretization of 1.

    Parameters
    ----------
    transmat : np.ndarray, shape=[N, N], dtype=float
        A reversible row stochastic transition matrix representing a
        discrete-time Markov chain.
    lag_time : float
        The time discretization interval for `transmat`. This means that
        discrete time Markov chain which `transmat` generates corresponds
        to some continuous time Markov process sampled at an interval of
        lag_time.
    pi : np.ndarray, shape=[N], dtype=float, optional
        The stationary distribution of transmat (which must exist since
        transmat must be reversible), satisfying `pi_i T_{ij} = pi_j T_{ji}`.
        If not supplied, `pi` can be computed.

    Notes
    -----
    The goal here is to find a transition matrix `T_1` such that
    [T_1]^lag_time == transmat. In general, the n-th root of a matrix is
    not unique, but here we have some constraints: we want T_1 to also be
    a reversible stochastic matrix.

    Furthermore, we postulate that transmat and T_1 both correspond to
    the transition matrix for some reversible continuous time Markov
    process sampled at intervals of 1 and lag_time respectively.
    [PROOF NEEDED] I think this implies that the eigenvalues of T_1 and
    transmat should both be 0 < lambda_i <= 1. With these constraints then,
    there is only 1 n-th root of transmat satisfying this property, which
    is computed by eigendecomposition transmat (here represented with `T_n`)
    as 

    .. math::

       T_n = U D U^{-1}
       T_1 = U D^{1/n} U^{-1}

    Since we have to cope with `transmat` possibly having negative eigenvalues,
    we just truncate negative eigenvalues to zero before taking the n-th root.
    Also note that the actual calculation does not require computing any matrix
    inverses, since you can first do a similarity transform of T_n into a real
    symmetrix matrix.
    """
    def stationary_eigenvector(transmat):
        # For large matrices, it would be more efficient to just
        # get the first eigenpair using scipy.sparse.linalg, but
        # for smaller matrices, scipy.linalg.eig getting all of
        # them seems faster
        eigvals, eigvecs = scipy.linalg.eig(transmat.T)
        # Equilibrium populations
        pi = eigvecs[:, eigvals.argmax()]
        pi /= pi.sum()
        return pi

    if pi is None:
        pi = stationary_eigenvector(transmat)

    # Because transmat is reversible, it's similar to a symmetric matrix.
    # by doing the calculation on the symmetric matrix, we avoid the need
    # to invert the eigenvectors when reconstructing the new transition
    # matrix from the eigendecomposition since the eigenvectors of the
    # symmetric matrix are orthogonal
    D = np.diag(pi ** 0.5)
    D_inv = np.diag(pi ** -0.5)
    tsym = D.dot(transmat).dot(D_inv)
    eigvals, eigvecs = scipy.linalg.eigh(tsym)

    contracted_eigvals = np.diag(np.maximum(eigvals, 0) ** (1.0 / float(lag_time)))
    result = D_inv.dot(eigvecs.dot(contracted_eigvals).dot(eigvecs.T)).dot(D)

    # this is a hack. we really need to constrain transmat to have positive
    # eigenvalues
    result = np.maximum(result, 0)
    return result / np.sum(result, axis=0)[np.newaxis, :]
