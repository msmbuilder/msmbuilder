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

    def h(X):
        c1 = np.sum((X[:block_dim, :block_dim] - B)**2)
        c2 = np.sum((X[:block_dim,block_dim:] - A)**2)
        c3 = np.sum((X[block_dim:,:block_dim] - A.T)**2)
        c4 = np.sum((X[block_dim:, block_dim:] - D)**2)
        return c1 + c2 + c3 + c4
    def gradh(X):
        grad1 = 2*(X[:block_dim, :block_dim] - B)
        grad2 = 2*(X[:block_dim,block_dim:] - A)
        grad3 = 2*(X[block_dim:,:block_dim] - A.T)
        grad4 = 2*(X[block_dim:, block_dim:] - D)

        grad = np.zeros((dim, dim))
        grad[:block_dim, :block_dim] = grad1
        grad[:block_dim,block_dim:] = grad2
        grad[block_dim:,:block_dim] = grad3
        grad[block_dim:, block_dim:] = grad4
        return grad

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
    for coord, mat in constraints:
        c = l2_batch_equals(X, mat, coord)
        sum_c += c
    return sum_c

def grad_many_batch_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for coord, mat in constraints:
        grad += grad_l2_batch_equals(X, mat, coord)
    return grad

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
    for c, P_coords, Q, R_coords in constraints:
        sum_c += batch_linear_equals(X, c, P_coords, Q, R_coords)
    return sum_c

def grad_many_batch_linear_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for c, P_coords, Q, R_coords in constraints:
        grad += grad_l2_batch_equals(X, c*get_entries(X, P_coords) + Q,
                    R_coords)
        if c != 0:
            grad += grad_l2_batch_equals(X,
                    (1./c)*get_entries(X, R_coords) - Q, P_coords)

    return grad

def Q_coords(dim):
    """
    Helper function that specifies useful coordinates for
    the Q convex program.
    """
    D_ADA_T_cds = (0, dim, 0, dim)
    I_1_cds = (0, dim, dim, 2*dim)
    I_2_cds = (dim, 2*dim, 0, dim)
    R_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    block_1_R_cds = (dim, 2*dim, dim, 2*dim)
    return (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds, block_1_R_cds)

def Q_constraints(dim, A, B, D):
    """
    Specifies the convex program required for Q optimization.

    minimize -log det R + Tr(RB)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    """
    We need to enforce zero equalities in X.
      -------------
     | -        -    0 |
C =  | -        _    0 |
     | 0        0    _ |
      -------------
    """
    constraints = [((2*dim, 3*dim, 0, 2*dim), np.zeros((dim, 2*dim))),
            ((0, 2*dim, 2*dim, 3*dim), np.zeros((2*dim, dim)))]
    """
    We need to enforce constant equalities in X.
      -------------
     |D-ADA.T   I    _ |
C =  | I        _    _ |
     | _        _    _ |
      -------------
    """
    D_ADA_T_cds = (0, dim, 0, dim)
    D_ADA_T = D - np.dot(A, np.dot(D, A.T))
    I_1_cds = (0, dim, dim, 2*dim)
    I_2_cds = (dim, 2*dim, 0, dim)
    constraints += [(D_ADA_T_cds, D_ADA_T), (I_1_cds, np.eye(dim)),
            (I_2_cds, np.eye(dim))]

    # Add constraints to Gs
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to enforce linear inequalities
          ----------
         |          |
    C =  |     R    |
         |        R |
          ----------
    """
    R_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    block_1_R_cds = (dim, 2*dim, dim, 2*dim)
    linear_constraints = [(1., R_cds, np.zeros((dim,dim)), block_1_R_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs
