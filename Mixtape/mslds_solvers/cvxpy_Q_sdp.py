import numpy as np
import scipy.linalg
import cvxpy as cvx
import mixtape.mslds_solvers.mslds_Q_sdp as Q_sdp

def cvx_Q_solve(dim, A, B, D):
    """
    Preconditions:

        1) B is a PSD matrix
        2) D is a PSD matrix
        3) D - A D A.T is PSD
    """
    # Numerical stability tranformations copied over from
    # CVXOPT implementation

    # Scale objective down by S for numerical stability
    eigs = np.linalg.eig(B)[0]
    # B may be a zero matrix (if no datapoints were associated here).
    S = max(abs(max(eigs)), abs(min(eigs)))
    if S != 0.:
        B = B / S
    else:
        B = B
    # Ensure that D doesn't have negative eigenvals
    # due to numerical issues
    min_D_eig = min(np.linalg.eig(D)[0])
    if min_D_eig < 0:
        # assume abs(min_D_eig) << 1
        D = D + 2 * abs(min_D_eig) * np.eye(x_dim)
    # Ensure that D - A D A.T is PSD. Otherwise, the problem is
    # unsolvable and weird numerical artifacts can occur.
    min_Q_eig = min(np.linalg.eig(D - np.dot(A, np.dot(D, A.T)))[0])
    if min_Q_eig < 0:
        # Scale A downwards until D - A D A.T is PSD
        eta = 0.99
        power = 1
        while (min(np.linalg.eig(D - np.dot((eta ** power) * A,
                               np.dot(D, (eta ** power) * A.T)))[0]) < 0):
            power += 1
        A = (eta ** power) * A

    # Compute intermediate quantities
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = np.real(scipy.linalg.sqrtm(B + epsilon * np.eye(dim)))
    Dinv = np.linalg.pinv(D)

    # Create two scalar variables.
    s = cvx.Variable(1)
    Z = cvx.semidefinite(dim, dim)
    Q = cvx.semidefinite(dim, dim)
    S1 = cvx.semidefinite(2*dim, 2*dim)
    S2 = cvx.semidefinite(2*dim, 2*dim)
    constraints = [
     #|Z+sI   F|
     #|F.T    Q| >= 0
     S1[:dim,:dim] == Z + s * np.eye(dim),
     S1[:dim,dim:] == F,
     S1[dim:,:dim] == F.T,
     S1[dim:,dim:] == Q,
     #|D-Q   A   |
     #|A.T D^{-1}| >= 0
     S2[:dim, :dim] == D - Q,
     S2[:dim, dim:] == A,
     S2[dim:, :dim] == A.T,
     S2[dim:, dim:] == Dinv]

    obj = cvx.Minimize(s * dim + sum([Z[i,i] for i in range(dim)]) -
            cvx.log_det(Q))
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return prob, s, Z, Q
