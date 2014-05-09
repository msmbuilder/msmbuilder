"""
Implementation of Hazan's algorithm

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.

for approximate solution of sparse semidefinite programs.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""

class BoundedTraceSolver(object):
    """
    Implementation of Hazan's Algorithm, which solves
    the optimization problem
         max f(X)
         X \in P
    where P = {X is PSD and Tr X = 1} is the set of PSD
    matrices with unit trace.
    NOTE: Should probably move f, gradf into constructor
    """
    def __init__(self):
        pass

    def solve(self, f, gradf, dim, N_iter, Cf=None, DEBUG=False,
                num_tries=5, alphas=None, X_init=None):
        """
        Parameters
        __________
        f: concave function
            Accepts (dim,dim) shaped matrices and outputs real
        gradf: function
            Computes grad f at given input matrix
        dim: int
            The dimensionality of the input vector space for f,
        N_iter: int
            The desired number of iterations
        Cf: float
            The curvature constant of function f (Optional).
        """
        v = random.rand(dim, 1)
        # orthonormalize v
        v = v / np.linalg.norm(v)
        if X_init == None:
            X = np.outer(v, v)
        else:
            X = np.copy(X_init)
        #X /= np.trace(X)
        import pdb
        pdb.set_trace()
        for j in range(N_iter):
            grad = gradf(X)
            print "\tIteration %d" % j
            if DEBUG:
                print "\tOriginal X:\n", X
                print "\tgrad X:\n", grad
            if dim >= 3:
                if Cf != None:
                    epsj = Cf/(j+1)
                else:
                    epsj = 1e-9
                # We usually try the following eigenvector finder,
                # which is based off an Implicitly Restarted
                # Arnoldi Method (essentially a stable version of
                # Lanczos's algorithm)
                try:
                    # shift matrices upwards by a positive quantity to
                    # avoid common issues with small eigenvalues
                    w, _ = linalg.eigsh(grad, k=1, tol=epsj, which='LM')
                    if np.isnan(w) or w == -np.inf or w == np.inf:
                        shift = 1
                    else:
                        shift = 1.5*np.abs(w)
                except (linalg.ArpackError, linalg.ArpackNoConvergence):
                    #print ("\tSmall eigenvalues leading to no " +
                    #         "convergence.  Shifting upwards.")
                    shift = 1
                vj = None
                last = 0.
                for i in range(num_tries):
                    try:
                        _, vj = linalg.eigsh(grad
                                + (i+1)*shift*np.eye(dim),
                                k=1, tol=epsj, which='LA')
                    except (linalg.ArpackError,
                            linalg.ArpackNoConvergence):
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
                            ws, vs = linalg.eigsh(grad
                                    + (i+1)*shift*np.eye(dim),
                                    k=k, tol=epsj, which='LA')
                        except (linalg.ArpackError,
                                linalg.ArpackNoConvergence):
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
            else:
                ws, vs = np.linalg.eig(grad)
                i = np.argmax(np.real(ws))
                vj = vs[:,i]

            # We don't really use Cf anymore...
            ## Avoid strange errors with complex numbers
            #vj = np.real(vj)
            #if alphas == None:
            #    alphaj = min(.5,2./(j+1))
            #else:
            #    alphaj = alphas[j]
            O = np.outer(vj, vj)
            step = (np.outer(vj,vj) - X)
            gamma = 1.0
            # Do simple back-tracking line search
            scale_down = 0.7
            f_X = f(X)
            gamma_best = gamma
            gamma_best_proj = gamma
            f_best = f((1.-gamma)*X + gamma*step)
            X_prop = (1 - gamma_best) * X + gamma_best*step
            f_best_proj = f_X
            N_tries = 30
            best_origin = None
            for count in range(N_tries):
                if f_best < f_X and count > N_tries:
                    break
                gamma = scale_down * gamma
                f_cur = f((1.-gamma)*X + gamma*step)
                if f_best < f_cur:
                    best_origin = 'Hazan'
                    f_best = f_cur
                    gamma_best = gamma
                    X_prop = (1.-gamma)*X + gamma*step
                # The following lines implement projected gradient
                # in the backtracking line search. This will be
                # inefficient for large matrices. Figure out a way
                # to get rid of it ......
                X_proj = X + gamma * grad
                X_proj = scipy.linalg.sqrtm(np.dot(X_proj.T, X_proj))
                f_cur_proj = f(X_proj)
                if f_best < f_cur_proj:
                    best_origin = 'Proj'
                    f_best = f_cur_proj
                    X_prop = X_proj
            if DEBUG:
                print "\tf(X):\n", f(X)
                print "\talphaj:\n", alphaj
                print "\tvk vk.T:\n", np.outer(vj,vj)
                print "\tstep:\n", step
            if f(X_prop) > f(X):
                X = X_prop
            print "\t\tgamma: ", gamma_best
            print "\t\t\tf(X): ", f(X)
            print "\t\t\tBest Origin: ", best_origin
            #X = X + alphaj * (np.outer(vj,vj) - X)
        import pdb
        pdb.set_trace()
        return X


