import numpy as np
import numpy.random
import mixtape.mslds_solvers.mslds_A_sdp as A_sdp
import mixtape.mslds_solvers.mslds_Q_sdp as Q_sdp
from mixtape.mslds_solvers.cvxpy_A_sdp import cvx_A_solve
from mixtape.mslds_solvers.cvxpy_Q_sdp import cvx_Q_solve
from mixtape.utils import reinflate_cvxopt
import time

def test_A_generate_constraints(dim):
    """
    Generate matrices B, C, E, D, Q at random to test
    solvers for Q.

    Parameters
    __________

    dim: int
        dimension of desired square matrices.
    """

    B = np.random.rand(dim, dim)
    # Generate C rank one
    b = np.random.rand(dim)
    s = np.random.rand(dim)
    C = np.outer(b,s)
    # Generate E PSD
    E = np.random.rand(dim,dim)
    E = np.dot(E.T, E)
    # Generate D PSD
    D = np.random.rand(dim, dim)
    D = np.dot(D.T, D)
    # Generate Q PSD such that D - Q is PSD
    Q = np.random.rand(dim, dim)
    Q = np.dot(Q.T, Q)
    eta = 0.99
    power = 0
    while (min(np.linalg.eig(D - (eta ** power) * Q)[0]) < 0):
        power += 1
    Q = (eta ** power) * Q
    return B, C, E, D, Q

def test_Q_generate_constraints(dim):
    """
    Generate matrices A, B, D at random to test
    solvers for Q.

    Parameters
    __________

    dim: int
        dimension of desired square matrices.
    """
    # Generate B PSD
    B = np.random.rand(dim, dim)
    B = np.dot(B.T, B)

    # Generate D PSD
    D = np.random.rand(dim, dim)
    D = np.dot(B.T, B)

    # Generate a matrix with max eigenvalue 1
    A = np.random.rand(dim, dim)
    A = A / max(np.linalg.eig(A)[0])

    # Scale A downwards until D - A D A.T is PSD
    eta = 0.99
    power = 0
    while (min(np.linalg.eig(D - np.dot((eta ** power) * A,
                           np.dot(D, (eta ** power) * A.T)))[0]) < 0):
        power += 1
    A = (eta ** power) * A

    return A, B, D

def cvxtest_A_solve_sdp(x_dims, num_tests):
    for num in range(num_tests):
        print "Random CVXPY A Test %d" % num
        for x_dim in x_dims:
            B, C, E, D, Q = test_A_generate_constraints(x_dim)
            start = time.clock()
            prob, s, Z, A = cvx_A_solve(x_dim, B, C, E, D, Q)
            elapsed = (time.clock() - start)
            print "\tCVXPY"
            print "\ttime elapsed:", elapsed
            print "\tstatus:", prob.status
            print "\toptimal value:", prob.value
            print "\toptimal s:", s.value
            print "\toptimal Z:\n", Z.value
            print "\toptimal A:\n", A.value
            print
        print

def test_A_solve_sdp(x_dims):
    max_iters = 100
    show_display = False
    for x_dim in x_dims:
        print "x_dim", x_dim
        B, C, E, D, Q = test_A_generate_constraints(x_dim)
        start = time.clock()
        sol, c, G, h = A_sdp.solve_A(x_dim, B, C, E, D, Q, max_iters,
                show_display)
        elapsed = (time.clock() - start)
        x = np.array(sol['x'])
        print "\tCVXOPT"
        print "\ttime elapsed:", elapsed
        print "\tstatus:", sol['status']
        print "\tprimal objective:", sol['primal objective']
        print "\tdual objective:", sol['dual objective']
        print "\toptimal s:", x[0]
        z = x[1:int(x_dim*(x_dim+1)/2+1)]
        print "\toptimal Z:\n", reinflate_cvxopt(x_dim, z)
        a = x[int(x_dim*(x_dim+1)/2+1):]
        A = np.reshape(a, (x_dim, x_dim), order='F')
        print "\toptimal A:\n", A
        print
    print

def cvxtest_Q_solve_sdp(x_dims, num_tests):
    for num in range(num_tests):
        print "Random CVXPY Q Test %d", num
        for x_dim in x_dims:
            print "x_dim", x_dim
            A, B, D = test_Q_generate_constraints(x_dim)
            start = time.clock()
            prob, s, Z, Q = cvx_Q_solve(x_dim, A, B, D)
            elapsed = (time.clock() - start)
            print "\tCVXPY"
            print "\ttime elapsed:", elapsed
            print "\tstatus:", prob.status
            print "\toptimal value:", prob.value
            print "\toptimal s:", s.value
            print "\tOptimal Z:\n", np.array(Z.value)
            print "\tOptimal Q:\n", np.array(Q.value)
            print
        print

def test_Q_solve_sdp(x_dims):
    max_iters = 100
    show_display = False
    for x_dim in x_dims:
        print "x_dim", x_dim
        A, B, D = test_Q_generate_constraints(x_dim)
        start = time.clock()
        sol, c, Gs, hs = Q_sdp.solve_Q(x_dim, A, B, D, max_iters,
                show_display)
        elapsed = (time.clock() - start)
        x = np.array(sol['x'])
        print "\tCVXOPT"
        print "\ttime elapsed:", elapsed
        print "\tstatus:", sol['status']
        print "\tprimal objective:", sol['primal objective']
        print "\tdual objective:", sol['dual objective']
        print "\toptimal s:", x[0]
        z = x[1:int(x_dim*(x_dim+1)/2+1)]
        Z = reinflate_cvxopt(x_dim, z)
        print "\toptimal Z:\n", Z
        q = x[int(x_dim*(x_dim+1)/2+1):]
        Q = reinflate_cvxopt(x_dim, q)
        print "\toptimal Q:\n", Q
        print
    print

def test1():
    x_dims = [1]
    num_tests = 10
    cvxtest_Q_solve_sdp(x_dims, num_tests)
    #test_Q_solve_sdp(x_dims)

def test2():
    x_dims = [1]
    num_tests = 30
    cvxtest_A_solve_sdp(x_dims, num_tests)
    #test_A_solve_sdp(x_dims)

if __name__ == "__main__":
    #test1()
    test2()
