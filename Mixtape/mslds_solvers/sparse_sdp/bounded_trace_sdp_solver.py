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
        # (stable version of Lanczos's algorithm)
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
            print("sparse.linalg.eigsh failed; going to np.linalg.eigh")
            ws, vs = np.linalg.eigh(grad)
            i = np.argmax(np.real(ws))
            vj = vs[:, i]
        return vj

    def project_spectrahedron(self, X, N_rounds=2):
        """
        Project X onto the spectrahedron { Y | Y is PSD and Tr(Y) == 1}
        """
        for r in N_rounds:
            X = scipy.linalg.sqrtm(np.dot(X.T, X))
            X = X / np.trace(X)
        return X

    def backtracking_line_search(self, X, gradX, stepX, c=1e-4,
                                    rho=0.7, N_tries =30):
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
        c: float
            Magnitude used in Armijo conditions
        scale: float
            Geometric factor used to shrink step size
        N_tries: int
            Number of tries
        """
        f, gradf = self.f, self.gradf
        alpha = 1.
        # Calculate ascent magnitude
        mag = np.trace(gradX.T, stepX)
        fX = f(X)
        for count in range(N_tries):
            X_cur = X + gamma*step
            f_cur = f(X_cur)
            if fX > f_cur + c*mag:
                break
            alpha = rho * alpha
        return X_cur, alpha

    def solve(self, N_iter, X_init=None, disp=True, debug=False, modes=[]):
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
            results = []
            if 'frank_wolfe' in modes:
                vj = self.rank_one_approximation(grad)
                step = np.outer(vj, vj)
                X_fw, gamma_fw = self.backtracking_line_search(X,
                        step, gamma_init=1., scale=0.7)
                results += [(X_fw, gamma_fw, 'frank_wolfe')]
            if 'projected_gradient' in modes:
                step = grad
                X_proj, gamma_proj = self.backtracking_line_search(X,
                        step, gamma_init=1., scale=0.7)
                X_proj = self.project_spectrahedron(X_proj)
                results += [(X_proj, gamma_proj, 'projected_gradient')]
            if disp:
                print "\tIteration %d" % j
                print "\t\tf(X): ", f(X)
                print "\t\t\tTr(X): ", np.trace(X)
                print "\t\t\tgamma: ", gamma
                #print "\t\t\tmethod: ", method
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
