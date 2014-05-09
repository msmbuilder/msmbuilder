class FeasibilitySDPSolver(object):
    """ Implementation of Hazan's Fast SDP feasibility, which uses
        the bounded trace PSD solver above to solve general SDPs.
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def feasibility_grad(self, X, As, bs, Cs, ds, Fs, gradFs, Gs,
            gradGs, eps):
        (dim, _) = np.shape(X)
        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale_full(m, n, p, q, eps)
        def gradf(X):
            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        return gradf(X)

    def feasibility_val(self, X, As, bs, Cs, ds, Fs, Gs, eps):
        (dim, _) = np.shape(X)
        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale(m, n, p, q, eps)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        return f(X)

    def feasibility_solve(self, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs,
            eps, dim, N_iter=None, X_init=None):
        """
        Solves feasibility problems of the type

        Feasibility of X
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X)  = d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) = 1

        by optimizing neg_max_penalty function
        TODO: Switch to log_sum_exp_penalty once numerically stable

        Parameters
        __________
        As: list
            inequality square (dim, dim) numpy.ndarray matrices
        bs: list
            inequality floats
        Cs: list
            equality square (dim, dim) numpy.ndarray matrices
        ds: list
            equality floats
        Fs: list
            convex inequalities
        Gs: list
            convex equalities
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        """

        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale_full(m, n, p, q, eps)
        if N_iter == None:
            N_iter = int(1./eps)
        # Need to swap in some robust theory about Cf
        fudge_factor = 1.0
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)

        #import pdb
        #pdb.set_trace()
        start = time.clock()
        X = self._solver.solve(f, gradf, dim, N_iter, X_init=X_init)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", X
        print "\tf(X) = %f" % (fX)
        SUCCEED = not (fX < -fudge_factor*eps)
        print "\tSUCCEED: " + str(SUCCEED)
        print "\tComputation Time (s): ", elapsed
        #import pdb
        #pdb.set_trace()
        return X, fX, SUCCEED

