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
    for coord, mat in constraints:
        (dim, _) = np.shape(mat)
        c2 = l2_batch_equals(X, mat, coord)
        sum_c += c2
    if dim > 4:
        dim = dim/4.
    return (1./dim**2) * sum_c

def grad_many_batch_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for coord, mat in constraints:
        (dim, _) = np.shape(mat)
        grad2 = grad_l2_batch_equals(X, mat, coord)
        grad += grad2
    if dim > 4:
        dim = dim/4.
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
    for c, P_coords, Q, R_coords in constraints:
        (dim, _) = np.shape(Q)
        sum_c += batch_linear_equals(X, c, P_coords, Q, R_coords)
    if dim > 4:
        dim = dim/4.
    return (1./dim**2) * sum_c

def grad_many_batch_linear_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for c, P_coords, Q, R_coords in constraints:
        (dim, _) = np.shape(Q)
        grad += grad_l2_batch_equals(X, c*get_entries(X, P_coords) + Q,
                    R_coords)
        if c != 0:
            grad += grad_l2_batch_equals(X,
                    (1./c)*(get_entries(X, R_coords) - Q), P_coords)

    if dim > 4:
        dim = dim/4.
    return (1./dim**2) * grad
