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
        # We usually try the following eigenvector finder,
        # which is based off an Implicitly Restarted
        # Arnoldi Method (essentially a stable version of
        # Lanczos's algorithm)
        try:
            # shift matrices upwards by a positive quantity to
            # avoid common issues with small eigenvalues
            w, _ = scipy.sparse.linalg.eigsh(grad, k=1,
                                        tol=epsj, which='LM')
            if np.isnan(w) or w == -np.inf or w == np.inf:
                shift = 1
            else:
                shift = 1.5*np.abs(w)
        except (scipy.sparse.linalg.ArpackError,
                scipy.sparse.linalg.ArpackNoConvergence):
            shift = 1
        vj = None
        last = 0.
        for i in range(num_tries):
            try:
                _, vj = scipy.sparse.linalg.eigsh(grad
                        + (i+1)*shift*np.eye(dim),
                        k=1, tol=epsj, which='LA')
            except (scipy.sparse.linalg.ArpackError,
                    scipy.sparse.linalg.ArpackNoConvergence):
                continue
            last = i
            if not np.isnan(np.min(vj)):
                break
        if vj == None or np.isnan(np.min(vj)):
            # The gradient is singular. In this case resort
            # to the more expensive, but more stable eigh method,
            # which is based on a divide and conquer approach
            # instead of Lanczos
            print("Iteration %d: Gradient is singular" % j)
            # Going to try next smallest singular value
            print "Looking for largest nonzero eigenvalue"
            vj = None
            for k in range(2,dim):
                try:
                    ws, vs = scipy.sparse.linalg.eigsh(grad
                            + (i+1)*shift*np.eye(dim),
                            k=k, tol=epsj, which='LA')
                except (scipy.sparse.linalg.ArpackError,
                        scipy.sparse.linalg.ArpackNoConvergence):
                    continue
                if not np.isnan(np.min(vs[:,k-1])):
                    vj = vs[:,k-1]
                    print "Picked %d-th eigenvalue" % k
                    break
            if vj == None:
                print "switching to divide and conquer"
                ws, vs = np.linalg.eigh(grad)
                i = np.argmax(np.real(ws))
                vj = vs[:, i]
        return vj

    def backtracking_line_search(self, X, step, gamma=1.,
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
                X_prop = (1. - gamma)*X + gamma*step
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
                X_prop = X_proj
                gamma_best = gamma
            gamma = scale * gamma
        return X_prop, method, gamma_best


    def solve(self, N_iter, X_init=None, disp=True,
            num_tries=5, debug=False,):
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
            if disp:
                print "\tIteration %d" % j
            if debug:
                print "\tOriginal X:\n", X
                print "\tgrad X:\n", grad
            vj = self.rank_one_approximation(grad)
            O = np.outer(vj, vj)
            step = (O - X)
            X_prop, method = back_tracking_line_search(X, step,
                    gamma=1., scale=0.7, N_tries=20)
            if debug:
                print "\t\tgamma: ", gamma_best
                print "\t\t\tf(X): ", f(X)
                print "\t\t\tBest Origin: ", method
            if f(X_prop) > f(X):
                X = X_prop
        return X
