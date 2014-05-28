import numpy as np
from utils import get_entries, set_entries

def simple_equality_constraint():
    """
    Generate constraints that specify the problem

        feasibility(X)
        subject to
          x_11 + 2 x_22 == 1.5

    """
    dim = 2
    As, bs = [], []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [1.5]
    Fs, gradFs, Gs, gradGs = [], [], [], []
    return dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def simple_equality_and_inequality_constraint():
    """
    Generate constraints that specify the problem

        feasbility(X)
        subject to
            x_11 + 2 x_22 <= 1
            x_11 + 2 x_22 + 2 x_33 == 5/3
            #Tr(X) = x_11 + x_22 + x_33 == 1
    """
    dim = 3
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
    Fs, gradFs, Gs, gradGs = [], [], [], []
    return dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def quadratic_inequality():
    """
    Generate constraints that specify the problem

        max penalty(X)
        subject to
            x_11^2 + x_22^2 <= .5
            Tr(X) = x_11 + x_22 == 1
    """
    dim = 2
    As, bs, Cs, ds = [], [], [], []
    def f(X):
        return X[0,0]**2 + X[1,1]**2 - 0.5
    def gradf(X):
        grad = np.zeros(np.shape(X))
        grad[0,0] = 2 * X[0,0]
        grad[1,1] = 2 * X[1,1]
        return grad
    Fs = [f]
    gradFs = [gradf]
    Gs, gradGs = [], []
    return dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def quadratic_equality():
    """
    Check that the bounded trace implementation can handle
    low-dimensional quadratic equalities

    We specify the problem

        feasibility(X)
        subject to
            x_11^2 + x_22^2 = 0.5
            Tr(X) = x_11 + x_22 == 1
    """
    dim = 2
    As, bs, Cs, ds, Fs, gradFs = [], [], [], [], [], []
    def g(X):
        return X[0,0]**2 + X[1,1]**2 - 0.5
    def gradg(X):
        grad = np.zeros(np.shape(X))
        grad[0,0] = 2 * X[0,0]
        grad[1,1] = 2 * X[1,1]
        return grad
    Gs = [g]
    gradGs = [gradg]
    return dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def stress_inequalities(dim):
    """
    Stress test the bounded trace solver for
    inequalities.

    With As and bs as below, we specify the problem

    max penalty(X)
    subject to
        x_ii <= 1/2n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with small entries
    for the first n-1 diagonal elements, but a large element (about 1/2)
    for the last element.
    """
    As = []
    for i in range(dim-1):
        Ai = np.zeros((dim,dim))
        Ai[i,i] = 1
        As.append(Ai)
    bs = []
    for i in range(dim-1):
        bi = 1./(2*dim)
        bs.append(bi)
    Cs, ds, Fs, gradFs, Gs, gradGs = [], [], [], [], [], []
    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def stress_equalities(dim):
    """
    Specify problem

    max penalty(X)
    subject to
        x_ii == 0, i < n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.
    """
    As, bs = [], []
    Cs = []
    for j in range(dim-1):
        Cj = np.zeros((dim,dim))
        Cj[j,j] = 1
        Cs.append(Cj)
    ds = []
    for j in range(dim-1):
        dj = 0.
        ds.append(dj)
    Fs, gradFs, Gs, gradGs = [], [], [], []
    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def stress_inequalities_and_equalities(dim):
    """
    Generate specification for the problem

    feasibility(X)
    subject to
        x_ij == 0, i != j
        x11
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.
    """
    tol = 1e-3
    As = []
    for j in range(1,dim-1):
        Aj = np.zeros((dim,dim))
        Aj[j,j] = 1
        As.append(Aj)
    bs = []
    for j in range(1,dim-1):
        bs.append(tol)
    Cs = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                Ci = np.zeros((dim,dim))
                Ci[i,j] = 1
                Cs.append(Ci)
    ds = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                dij = 0.
                ds.append(dij)
    Fs, gradFs, Gs, gradGs = [], [], [], []
    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def basic_batch_equality(dim, A, B, D):
    """
    Explicity generates specification for the problem

    feasibility(X)
    subject to
        [[ B   , A],
         [ A.T , D]]  is PSD, where B, D are arbitrary, A given.

        Tr(X) = Tr(B) + Tr(D) == 1
    """
    As, bs, Cs, ds, Fs, gradFs = [], [], [], [], [], []
    block_dim = int(dim/2)

    B_cds = (0, block_dim, 0, block_dim)
    A_cds = (0, block_dim, block_dim, 2*block_dim)
    A_T_cds = (block_dim, 2*block_dim, 0, block_dim)
    D_cds = (block_dim, 2*block_dim, block_dim, 2*block_dim)
    constraints = [(B_cds, B), (A_cds, A), (A_T_cds, A.T), (D_cds, D)]
    def h(X):
        return many_batch_equals(X, constraints)
    def gradh(X):
        return grad_many_batch_equals(X, constraints)

    Gs = [h]
    gradGs = [gradh]
    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def l1_batch_equals(X, A, coord):
    c = np.sum(np.abs(get_entries(X,coord) - A))
    return c

def grad_l1_batch_equals(X, A, coord):
    # Upper right
    grad_piece = np.sign(get_entries(X,coord) - A)
    grad = np.zeros(np.shape(X))
    set_entries(grad, coord, grad_piece)
    return grad

def l2_batch_equals(X, A, coord):
    c = np.sum((get_entries(X,coord) - A)**2)
    return c

def grad_l2_batch_equals(X, A, coord):
    # Upper right
    grad_piece = 2*(get_entries(X,coord) - A)
    grad = np.zeros(np.shape(X))
    set_entries(grad, coord, grad_piece)
    return grad

def many_batch_equals(X, constraints):
    sum_c = 0
    (dim, _) = np.shape(X)
    for coord, mat in constraints:
        c2 = l2_batch_equals(X, mat, coord)
        sum_c += c2
    return (1./dim**2) * sum_c

def grad_many_batch_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    (dim, _) = np.shape(X)
    for coord, mat in constraints:
        grad2 = grad_l2_batch_equals(X, mat, coord)
        grad += grad2
    return (1./dim**2) * grad

def batch_linear_equals(X, c, P_coords, Q, R_coords):
    """
    Performs operation R_coords = c * P_coords + Q
    """
    val = l2_batch_equals(X, c*get_entries(X, P_coords) + Q, R_coords)
    return val

def grad_batch_linear_equals(X, c, P_coords, Q, R_coords):
    grad = np.zeros(np.shape(X))
    grad += grad_l2_batch_equals(X, c*get_entries(X, P_coords) + Q,
            R_coords)
    if c != 0:
        grad += grad_l2_batch_equals(X,(1./c)*get_entries(X, R_coords) - Q,
                                    P_coords)
    return grad

def many_batch_linear_equals(X, constraints):
    sum_c = 0
    (dim, _) = np.shape(X)
    for c, P_coords, Q, R_coords in constraints:
        sum_c += batch_linear_equals(X, c, P_coords, Q, R_coords)
    return (1./dim**2) * sum_c

def grad_many_batch_linear_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    (dim, _) = np.shape(X)
    for c, P_coords, Q, R_coords in constraints:
        grad += grad_l2_batch_equals(X, c*get_entries(X, P_coords) + Q,
                    R_coords)
        if c != 0:
            grad += grad_l2_batch_equals(X,
                    (1./c)*(get_entries(X, R_coords) - Q), P_coords)

    return (1./dim**2) * grad

def A_coords(dim):
    """
      ----------------------
     | _        _    0   0 |
     | _        _    0   0 |
     | 0        0    _   _ |
     | 0        0    _   _ |
      ----------------------
    """
    # Should there be coordinates for the zeros?
    """
      ---------------------
     |D-Q       _    _   _ |
     | _     D^{-1}  _   _ |
     | _        _    I   _ |
     | _        _    _   I |
      ---------------------
    """
    D_Q_cds = (0, dim, 0, dim)
    Dinv_cds = (dim, 2*dim, dim, 2*dim)
    I_1_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    I_2_cds = (3*dim, 4*dim, 3*dim, 4*dim)


    """
      --------------------
     |  _     A     _   _ |
     | A.T    _     _   _ |
     |  _     _     _   A |
     |  _     _    A.T  _ |
      --------------------
    """
    A_1_cds = (0, dim, dim, 2*dim)
    A_T_1_cds = (dim, 2*dim, 0, dim)
    A_2_cds = (2*dim, 3*dim, 3*dim, 4*dim)
    A_T_2_cds = (3*dim, 4*dim, 2*dim, 3*dim)

    return (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
            A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds)

def A_constraints(block_dim, D, Dinv, Q, mu, stability=False):

    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(block_dim)

    """
    We need to enforce zero equalities in X
      ----------------------
     | _        _    0   0 |
C =  | _        _    0   0 |
     | 0        0    _   _ |
     | 0        0    _   _ |
      ----------------------
    """
    constraints = [((2*block_dim, 4*block_dim, 0, 2*block_dim),
                    np.zeros((2*block_dim, 2*block_dim))),
                   ((0, 2*block_dim, 2*block_dim, 4*block_dim),
                    np.zeros((2*block_dim, 2*block_dim)))]

    """
    We need to enforce constant equalities in X
      ---------------------
     |D-Q       _    _   _ |
C =  | _     D^{-1}  _   _ |
     | _        _    I   _ |
     | _        _    _   I |
      ---------------------
    """
    D_Q = D-Q
    constraints += [(D_Q_cds, D_Q), (Dinv_cds, Dinv),
            (I_1_cds, np.eye(block_dim)), (I_2_cds, np.eye(block_dim))]

    # Add constraints to Gs
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """
    We need to enforce linear inequalities

          --------------------
         |  _     A     _   _ |
    X =  | A.T    _     _   _ |
         |  _     _     _   A |
         |  _     _    A.T  _ |
          --------------------
    """
    linear_constraints = [(1., A_1_cds, np.zeros((block_dim, block_dim)),
                            A_2_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    if stability:
        """
        We need to enforce stability constraint

        A mu == 0

        We thus place constraint ||A mu||^2 = mu^T A^T A mu == 0

        The gradient is matrix

               [[ mu.T ]
                [ mu.T ]
        G =       ...
                [ mu.T ]]
        """
        mu = np.reshape(mu, (block_dim, 1))
        def stability(X):
            (dim, _) = np.shape(X)
            A_1 = get_entries(X, A_1_cds)
            diff1 = np.dot(A_1, mu)

            A_T_1 = get_entries(X, A_T_1_cds)
            diffT1 = np.dot(A_T_1.T, mu)

            A_2 = get_entries(X, A_2_cds)
            diff2 = np.dot(A_2, mu)

            A_T_2 = get_entries(X, A_T_2_cds)
            diffT2 = np.dot(A_T_2.T, mu)

            return (1./dim**2) * (np.dot(diff1.T, diff1)
                                + np.dot(diffT1.T, diffT1)
                                + np.dot(diff2.T, diff2)
                                + np.dot(diffT2.T, diffT2))

        def grad_stability(X):
            (dim, _) = np.shape(X)
            G = np.zeros(np.shape(X))
            A_1 = get_entries(X, A_1_cds)
            diff1 = 2 * np.dot(A_1, mu)
            grad1 = np.tile(mu.T, (block_dim, 1))
            # (block_dim,1) * (block_dim,block_dim) across rows
            grad1 = diff1 * grad1
            set_entries(G, A_1_cds, grad1)

            A_T_1 = get_entries(X, A_T_1_cds)
            diffT1 = 2 * np.dot(A_T_1.T, mu)
            gradT1 = np.tile(mu.T, (block_dim, 1))
            gradT1 = diffT1 * gradT1
            set_entries(G, A_T_1_cds, gradT1.T)

            A_2 = get_entries(X, A_2_cds)
            diff2 = 2 * np.dot(A_2, mu)
            grad2 = np.tile(mu.T, (block_dim, 1))
            grad2 = diff2 * grad2
            set_entries(G, A_2_cds, grad2)

            A_T_2 = get_entries(X, A_T_2_cds)
            diffT2 = 2 * np.dot(A_T_2.T, mu)
            gradT2 = np.tile(mu.T, (block_dim, 1))
            gradT2 = diffT2 * gradT2
            set_entries(G, A_T_2_cds, gradT2.T)
            return (1./dim**2) * G
        Gs.append(stability)
        gradGs.append(grad_stability)

    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def Q_coords(dim):
    """
    Helper function that specifies useful coordinates for
    the Q convex program.
    minimize -log det R + Tr(RB)
          -------------------
         |D-ADA.T  I         |
    X =  |   I     R         |
         |            D   cI |
         |           cI   R  |
          -------------------
    X is PSD
    """
    # Block 1
    D_ADA_T_cds = (0, dim, 0, dim)
    I_1_cds = (0, dim, dim, 2*dim)
    I_2_cds = (dim, 2*dim, 0, dim)
    R_1_cds = (dim, 2*dim, dim, 2*dim)

    # Block 2
    D_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    c_I_1_cds = (2*dim, 3*dim, 3*dim, 4*dim)
    c_I_2_cds = (3*dim, 4*dim, 2*dim, 3*dim)
    R_2_cds = (3*dim, 4*dim, 3*dim, 4*dim)
    return (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
                D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) 

def Q_constraints(dim, A, B, D, c):
    """
    Specifies the convex program required for Q optimization.

    minimize -log det R + Tr(RB)
          -------------------
         |D-ADA.T  I         |
    X =  |   I     R         |
         |            D   cI |
         |           cI   R  |
          -------------------
    X is PSD
    """

    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
        D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
            Q_coords(dim)

    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    """
    We need to enforce zero equalities in X.
      ---------------
     | _  _    0  0  |
C =  | _  _    0  0  |
     | 0  0    _  _  |
     | 0  0    _  _  |
      ---------------
    """
    constraints = [((2*dim, 4*dim, 0, 2*dim), np.zeros((2*dim, 2*dim))),
            ((0, 2*dim, 2*dim, 4*dim), np.zeros((2*dim, 2*dim)))]
    """
    We need to enforce constant equalities in X.
      ---------------------
     |D-ADA.T   I    _  _  |
C =  | I        _    _  _  |
     | _        _    D  cI |
     | _        _   cI  _  |
      ---------------------
    """
    D_ADA_T = D - np.dot(A, np.dot(D, A.T))
    constraints += [(D_ADA_T_cds, D_ADA_T), (I_1_cds, np.eye(dim)),
                    (I_2_cds, np.eye(dim)), (D_cds, D),
                    (c_I_1_cds, c*np.eye(dim)), (c_I_2_cds, c*np.eye(dim))]

    # Add constraints to Gs
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to enforce linear inequalities
          -----------
         |-  -       |
    C =  |-  R  -  - |
         |      -  R |
          -----------
    """
    linear_constraints = [(1., R_1_cds, np.zeros((dim,dim)), R_2_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs
