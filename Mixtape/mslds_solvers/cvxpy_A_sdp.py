import numpy as np
import scipy.linalg
import cvxpy as cvx
import mixtape.mslds_solvers.mslds_A_sdp as A_sdp

def cvx_A_solve(dim, B, C, E, D, Q):
    """
    Inputs should satisfy the following properties

    1) E is PD
    2) Q is PD
    3) D is PD
    4) C is a rank one matrix
    4) D - Q is PSD
    """

    # Numerical stability tranformations copied over from
    # CVXOPT implementation

    # Scale input matrices down by S (see below) for numerical stability
    eigsQinv = max([abs(1. / q) for q in np.linalg.eig(Q)[0]])
    eigsE = max([abs(e) for e in np.linalg.eig(E)[0]])
    eigsCB = max([abs(cb) for cb in np.linalg.eig(C - B)[0]])
    S = max(eigsQinv, eigsE, eigsCB)
    Q = Q / S
    E = E / S
    C = C / S
    B = B / S
    # Ensure that D doesn't have negative eigenvals
    # due to numerical issues
    min_D_eig = min(np.linalg.eig(D)[0])
    if min_D_eig < 0:
        # assume abs(min_D_eig) << 1
        D = D + 2 * abs(min_D_eig) * np.eye(x_dim)

    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    J = np.real(scipy.linalg.sqrtm(scipy.linalg.pinv2(Q)
                    + epsilon * np.eye(dim)))
    H = np.real(scipy.linalg.sqrtm(E + epsilon * np.eye(dim)))
    F = np.dot(J, C - B)
    Dinv = np.linalg.pinv(D)

    # Create scalar variables
    s = cvx.Variable(1)
    Z = cvx.semidefinite(dim, dim)
    A = cvx.Variable(dim, dim)
    S1 = cvx.semidefinite(2*dim, 2*dim)
    S2 = cvx.semidefinite(2*dim, 2*dim)
    S3 = cvx.semidefinite(2*dim, 2*dim)

    eps = 1e-4 # Is this necessary?
    constraints = [
    #|Z+sI-JAF.T -FA.TJ  JAH|
    #|    (JAH).T         I |
    S1[:dim,:dim] == Z + s*np.eye(dim) - J*A*F.T - F*A.T*J,
    S1[:dim,dim:] == J*A*H,
    S1[dim:,:dim] == H.T*A.T*J.T,
    S1[dim:,dim:] == np.eye(dim),
    #|D-eps_I    A      |
    #|A.T        D^{-1} |
    S2[:dim,:dim] == D - eps * D,
    S2[:dim,dim:] == A,
    S2[dim:,:dim] == A.T,
    S2[dim:,dim:] == Dinv,
    #|I  A.T|
    #|A   I |
    S3[:dim,:dim] == np.eye(dim),
    S3[:dim,dim:] == A.T,
    S3[dim:,:dim] == A,
    S3[dim:,dim:] == np.eye(dim)
    ]

    obj = cvx.Minimize(s * dim + sum([Z[i,i] for i in range(dim)]))
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return prob, s, Z, A
