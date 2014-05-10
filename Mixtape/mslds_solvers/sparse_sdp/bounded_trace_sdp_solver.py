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

    def project_spectrahedron(self, X, N_rounds=4):
        """
        Project X onto the spectrahedron { Y | Y is PSD and Tr(Y) == 1}
        """
        for r in range(N_rounds):
            # Project onto semidefinite cone
            Z = np.zeros(np.shape(X))
            ws, vs = np.linalg.eigh(Z)
            for i in range(len(ws)):
                w = ws[i]
                if w >= 0:
                    Z += max(w, 0) * np.outer(vs[:, i], vs[:, i])
            X = Z
            # Project onto tr = 1 plane
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
        gradX: np.ndarray
            Gradient at X
        stepX: np.ndarray
            Search direction
        c: float
            Magnitude used in Armijo conditions
        rho: float
            Geometric factor used to shrink step size
        N_tries: int
            Number of tries
        """
        f, gradf = self.f, self.gradf
        alpha = 1.
        # Calculate ascent magnitude
        mag = np.trace(np.dot(gradX.T, stepX))
        fX = f(X)
        for count in range(N_tries):
            X_cur = X + alpha*stepX
            fX_cur = f(X_cur)
            if fX_cur > fX + c*mag:
                break
            alpha = rho * alpha
        return fX_cur, X_cur, alpha

    def solve(self, N_iter, X_init=None, disp=True, debug=False,
            methods=[]):
        """
        Parameters
        __________
        N_iter: int
            The desired number of iterations
        """
        import pdb
        pdb.set_trace()
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
            if 'frank_wolfe' in methods:
                vj = self.rank_one_approximation(grad)
                O = np.outer(vj, vj)
                step = O - X
                fX_fw, X_fw, alpha_fw = \
                        self.backtracking_line_search(X, grad, step)
                results += [(fX_fw, X_fw, alpha_fw, 'frank_wolfe')]
            if 'projected_gradient' in methods:
                step = grad
                fX_proj, X_proj, alpha_proj = \
                        self.backtracking_line_search(X, grad, step)
                X_proj = self.project_spectrahedron(X_proj)
                results += [(fX_proj, X_proj, alpha_proj,
                                'projected_gradient')]
            ind = np.argmax(np.array([result[0] for result in results]))
            fX_prop,  X_prop, alpha, method = results[ind]
            if disp:
                print "\tIteration %d" % j
                print "\t\tf(X): ", f(X)
                print "\t\t\tTr(X): ", np.trace(X)
                print "\t\t\talpha: ", alpha
                print "\t\t\tmethod: ", method
            if debug:
                print "X\n", X
                print "grad\n", grad
            if fX_prop > f(X):
                X = X_prop
            elif np.array_equal(X, X_prop):
                # We're stuck in fixed point.
                break
        return X
