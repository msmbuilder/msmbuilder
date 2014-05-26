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

    def rank_one_approximation(self, grad, disp=True):
        epsj = 1e-9
        # Use Implicitly Restarted Arnoldi Method (IRAM)
        # (stable version of Lanczos's algorithm)
        shifts = [1., 10., 100., 1000.]
        vj = None
        stable = False
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
            if disp:
                print("sparse.linalg.eigsh failed; "
                      "going to np.linalg.eigh")
            ws, vs = np.linalg.eigh(grad)
            i = np.argmax(np.real(ws))
            vj = vs[:, i]
            stable = True
        return vj, stable

    def stable_rank_one_approximation(self, grad):
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
            ws, vs = np.linalg.eigh(X)
            for i in range(len(ws)):
                w = ws[i]
                if w >= 0:
                    Z += max(w, 0) * np.outer(vs[:, i], vs[:, i])
            X = Z
            # Project onto tr = 1 plane
            X = X / np.trace(X)
        return X

    def backtracking_line_search(self, X, gradX, stepX, c=1e-4,
                                    rho=0.7, N_tries=30):
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
            methods=['frank_wolfe'], early_exit=True, min_step_size=1e-6,
            good_enough=None, num_stable=np.inf):
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
        fX = f(X)
        stable_so_far = 0
        for j in range(N_iter):
            grad = gradf(X)
            results = []
            if good_enough != None:
                if fX >= good_enough:
                    break
            if stable_so_far > num_stable:
                break
            if 'frank_wolfe' in methods:
                vj, stable = self.rank_one_approximation(grad, disp=disp)
                if stable:
                    stable_so_far += 1
                O = np.outer(vj, vj)
                step = O - X
                fX_fw, X_fw, alpha_fw = \
                        self.backtracking_line_search(X, grad, step)
                results += [(fX_fw, X_fw, alpha_fw, 'frank_wolfe')]
            if 'frank_wolfe_stable' in methods:
                vj = self.stable_rank_one_approximation(grad)
                O = np.outer(vj, vj)
                step = O - X
                fX_fw, X_fw, alpha_fw = \
                        self.backtracking_line_search(X, grad, step)
                results += [(fX_fw, X_fw, alpha_fw, 'frank_wolfe_stable')]
            ind = np.argmax(np.array([result[0] for result in results]))
            fX_prop,  X_prop, alpha, method = results[ind]
            delta = 0
            if (early_exit and
                    (fX_prop <= fX + min_step_size 
                        or np.sum(np.abs(X_prop - X)) < min_step_size)):
                delta = fX_prop - fX
                if disp:
                    print "\t\t\tdelta: ", delta
                    print "\t\t\tEarly Stopping."
                break
            elif fX_prop > fX:
                delta = fX_prop - fX
                X = X_prop
                fX = fX_prop
            if disp:
                print "\tIteration %d" % j
                print "\t\tf(X): ", f(X)
                print "\t\t\tdelta: ", delta
                print "\t\t\tTr(X): ", np.trace(X)
                print "\t\t\talpha: ", alpha
                print "\t\t\tmethod: ", method
            if disp and debug:
                print "X\n", X
                print "grad\n", grad
        return X
