import numpy as np

def simple_equality_constraint():
    """
    Generate constraints that specify the problem

        feasibility(X)
        subject to
          x_11 + 2 x_22 == 1.5
          Tr(X) = x_11 + x_22 == 1.

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
            Tr(X) = x_11 + x_22 + x_33 == 1
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
    def g(X):
        c1 = np.sum(np.abs(X[:block_dim, :block_dim] - B))
        c2 = np.sum(np.abs(X[:block_dim,block_dim:] - A))
        c3 = np.sum(np.abs(X[block_dim:,:block_dim] - A.T))
        c4 = np.sum(np.abs(X[block_dim:, block_dim:] - D))
        return c1 + c2 + c3 + c4
    def gradg(X):
        grad1 = np.sign(X[:block_dim, :block_dim] - B)
        grad2 = np.sign(X[:block_dim,block_dim:] - A)
        grad3 = np.sign(X[block_dim:,:block_dim] - A.T)
        grad4 = np.sign(X[block_dim:, block_dim:] - D)

        grad = np.zeros((dim, dim))
        grad[:block_dim, :block_dim] = grad1
        grad[:block_dim,block_dim:] = grad2
        grad[block_dim:,:block_dim] = grad3
        grad[block_dim:, block_dim:] = grad4
        return grad

    Gs = [g]
    gradGs = [gradg]
    return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def batch_equals(X, A, x_low, x_hi, y_low, y_hi):
    c = np.sum(np.abs(X[x_low:x_hi,y_low:y_hi] - A))
    return c

def grad_batch_equals(X, A, x_low, x_hi, y_low, y_hi):
    # Upper right
    grad_piece = np.sign(X[x_low:x_hi,y_low:y_hi] - A)
    grad = np.zeros(np.shape(X))
    grad[x_low:x_hi,y_low:y_hi] = grad_piece
    return grad

Scale = 2.0
L1Scale = 0.05
def many_batch_equals(X, constraints):
    sum_c = 0
    for coord, mat in constraints:
        #c = np.sum(np.abs(get_entries(X, coord) - mat))
        c = np.sum((get_entries(X, coord) - mat)**2)
        c += L1Scale * np.sum(np.abs(get_entries(X, coord) - mat))
        sum_c += c
    return Scale * sum_c

def grad_many_batch_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for coord, mat in constraints:
        #grad_piece = np.sign(get_entries(X, coord) - mat)
        grad_piece = 2*(get_entries(X, coord) - mat)
        grad_piece += L1Scale * np.sign(get_entries(X, coord) - mat)
        set_entries(grad, coord, grad_piece)
    return Scale * grad

def batch_linear_equals(X, c, P_coords, Q, R_coords):
    """
    Performs operation R_coords = c * P_coords + Q
    """
    #c = np.sum(np.abs(c * get_entries(X, P_coords) + Q
    #                - get_entries(X, R_coords)))
    c += np.sum(np.abs(c * get_entries(X, P_coords) + Q
                    - get_entries(X, R_coords)))
    c += L1Scale * np.sum((c * get_entries(X, P_coords) + Q
                    - get_entries(X, R_coords))**2)
    return c

def many_batch_linear_equals(X, constraints):
    sum_c = 0
    for c, P_coords, Q, R_coords in constraints:
        #c = np.sum(np.abs(c * get_entries(X, P_coords) + Q
        #                - get_entries(X, R_coords)))
        c = np.sum((c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords))**2)
        c += L1Scale * np.sum(np.abs(c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords)))
        sum_c += c
    return sum_c

def grad_batch_linear_equals(X, c, P_coords, Q, R_coords):
    grad = np.zeros(np.shape(X))
    #grad_piece_P = c * np.sign(c * get_entries(X, P_coords) + Q
    #                    - get_entries(X, R_coords))
    #grad_piece_R = - np.sign(c * get_entries(X, P_coords) + Q
    #                    - get_entries(X, R_coords))
    grad_piece_P = c * 2*(c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords))
    grad_piece_R = - 2*(c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords))
    grad_piece_P += L1Scale * c * np.sign(c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords))
    grad_piece_R += L1Scale * - np.sign(c * get_entries(X, P_coords) + Q
                        - get_entries(X, R_coords))
    set_entries(grad, P_coords, grad_piece_P)
    set_entries(grad, R_coords, grad_piece_R)
    return grad

def grad_many_batch_linear_equals(X, constraints):
    grad = np.zeros(np.shape(X))
    for c, P_coords, Q, R_coords in constraints:
        #grad_piece_P = c * np.sign(c * get_entries(X, P_coords) + Q
        #                    - get_entries(X, R_coords))
        #grad_piece_R = - np.sign(c * get_entries(X, P_coords) + Q
        #                    - get_entries(X, R_coords))
        grad_piece_P = c * 2*(c * get_entries(X, P_coords) + Q
                            - get_entries(X, R_coords))
        grad_piece_R = - 2*(c * get_entries(X, P_coords) + Q
                            - get_entries(X, R_coords))
        grad_piece_P += L1Scale*c*np.sign(c * get_entries(X, P_coords) + Q
                            - get_entries(X, R_coords))
        grad_piece_R += L1Scale * -np.sign(c * get_entries(X, P_coords) + Q
                            - get_entries(X, R_coords))
        set_entries(grad, P_coords, grad_piece_P)
        set_entries(grad, R_coords, grad_piece_R)
    return grad

