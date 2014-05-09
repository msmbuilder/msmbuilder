"""
Implementation of Hazan's algorithm

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.

for approximate solution of sparse semidefinite programs.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

class BoundedTraceSolver(object):
    """
    Implementation of Hazan's Algorithm, which solves
    the optimization problem
         max f(X)
         X \in P
    where P = {X is PSD and Tr X = 1} is the set of PSD
    matrices with unit trace.
    """
    def __init__(self, f, gradf, dim):
        """
        Arguments
        _________
        f: concave function
            Accepts (dim,dim) shaped matrices and outputs real
        gradf: function
            Computes grad f at given input matrix
        dim: int
            The dimensionality of input matrices
        """
        self.f = f
        self.gradf = gradf
        self.dim = dim

    def rank_one_approximation(self, grad):
        epsj = 1e-9
        # Use Implicitly Restarted Arnoldi Method (IRAM)
        # (essentially a stable version of Lanczos's algorithm)
        shifts = [1., 10., 100., 1000.]
        vj = None
        for shift in shifts:
            try:
                _, vj = scipy.sparse.linalg.eigsh(grad
                        + shift*np.eye(self.dim),
                        k=1, tol=epsj, which='LA')
            except (scipy.sparse.linalg.ArpackError,
                    scipy.sparse.linalg.ArpackNoConvergence):
                continue
            if not np.isnan(np.min(vj)):
                break
        if vj == None or np.isnan(np.min(vj)):
            # IRAM failed. Use np.linalg.eigh
            print("sparse.linalg.eigsh failed; going to np.linalg.eigh")
            # Going to try next smallest singular value
            ws, vs = np.linalg.eigh(grad)
            i = np.argmax(np.real(ws))
            vj = vs[:, i]
        return vj

    def backtracking_line_search(self, X, step, grad, gamma=1.,
                                    scale=0.7, N_tries =30):
        """
        Implements simple backtracking line search for hill-climbing.
        Parameters
        __________

        X: np.ndarray
            Current position
        step: np.ndarray
            Search direction
        gamma: float
            Initial step size
        scale: float
            Geometric factor used to shrink step size
        N_tries: int
            Number of tries
        """
        f, gradf = self.f, self.gradf
        # Do simple back-tracking line search
        X_best = X
        f_best = f(X_best)
        method = None
        gamma_best = gamma
        for count in range(N_tries):
            X_cur = (1. - gamma)*X + gamma*step
            f_cur = f(X_cur)
            if f_best < f_cur:
                method = 'Hazan'
                f_best = f_cur
                X_best = X_cur
                gamma_best = gamma
            # The following lines implement projected gradient
            # in the backtracking line search. This will be
            # inefficient for large matrices. Figure out a way
            # to get rid of it ......
            X_cur_proj = X + gamma * grad
            X_cur_proj = scipy.linalg.sqrtm(
                            np.dot(X_cur_proj.T, X_cur_proj))
            X_cur_proj = X_cur_proj / np.trace(X_cur_proj)
            f_cur_proj = f(X_cur_proj)
            if f_best < f_cur_proj:
                method = 'Proj'
                f_best = f_cur_proj
                X_best = X_cur_proj
                gamma_best = gamma
            gamma = scale * gamma
        return X_best, method, gamma_best

    def solve(self, N_iter, X_init=None, disp=True, debug=False,):
        """
        Parameters
        __________
        N_iter: int
            The desired number of iterations
        """
        f, gradf = self.f, self.gradf
        if X_init == None:
            v = np.random.rand(self.dim, 1)
            # orthonormalize v
            v = v / np.linalg.norm(v)
            X = np.outer(v, v)
        else:
            X = np.copy(X_init)
        for j in range(N_iter):
            grad = gradf(X)
            vj = self.rank_one_approximation(grad)
            step = np.outer(vj, vj)
            X_prop, method, gamma = self.backtracking_line_search(X,
                    step, grad, gamma=1., scale=0.7, N_tries=20)
            if disp:
                print "\tIteration %d" % j
                print "\t\tf(X): ", f(X)
                print "\t\t\tTr(X): ", np.trace(X)
                print "\t\t\tgamma: ", gamma
                print "\t\t\tmethod: ", method
            if debug:
                print "X\n", X
                print "grad\n", grad
            if f(X_prop) > f(X):
                X = X_prop
            else:
                if np.array_equal(X, X_prop):
                    # We're stuck in fixed point.
                    break
        return X
