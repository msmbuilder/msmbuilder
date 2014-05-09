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


def batch_equals(X, A, x_low, x_hi, y_low, y_hi):
    c = np.sum(np.abs(X[x_low:x_hi,y_low:y_hi] - A))
    return c

def batch_equals_grad(X, A, x_low, x_hi, y_low, y_hi):
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

