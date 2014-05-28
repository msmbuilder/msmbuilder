import numpy as np
from penalties import *
from bounded_trace_sdp_solver import BoundedTraceSolver


class FeasibilitySolver(object):
    """ Solves optimization problem

        feasibility(X)
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X) == d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) <= R

        We tranform this problem into a form solvable by the bounded trace
        solver.  We normalize the trace upper bound Tr(X) <= R to Tr(X) <=
        1 by performing change of variable

            X := X / R

        We then perform scalings

            A_i         := A_i * R
            C_i         := C_i * R
            f_k(X)      := f_k(R * X)
            grad f_k(X) := (1/R) * grad g_k(R * X)
            g_l(X)      := g_l(R * X)
            grad g_l(X) := (1/R) * grad g_l(R * X)

        To transform inequality Tr(X) <= 1 to Tr(X) == 1, we perform
        change of variable

        Y  := [[X,    0     ],
               [0, 1 - tr(X)]]

        Y is PSD if and only if X is PSD and y is real and nonnegative.
        The constraint that Tr Y = 1 is true if and only if Tr X <= 1.
        We transform the origin constraints by

         A_i := [[A_i, 0],  C_i := [[C_i, 0],
                 [ 0,  0]]          [ 0,  0]]

        f_k(Y)      := f_k(X)
        grad f_k(Y) := [[ grad f_k(X), 0], # Multiply this by 1/R?
                        [      0     , 0]]
        g_l(Y)      := g_l(X)
        grad g_l(Y) := [[ grad g_l(X), 0], # Multiply this by 1/R?
                        [      0     , 0]]

    """
    def __init__(self, dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
        self.dim = dim
        self.eps = eps
        (self.As, self.bs, self.Cs, self.ds, self.Fs, self.gradFs,
                self.Gs, self.gradGs) = \
                        (As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)

    def init_solver(self, R):
        (Aprimes, bprimes, Cprimes, dprimes, Fprimes, gradFprimes,
                Gprimes, gradGprimes) = \
                    self.transform_input(R, self.As, self.bs, self.Cs,
                            self.ds, self.Fs, self.gradFs,
                            self.Gs, self.gradGs)
        # Perhaps get rid of these?
        (self._Aprimes, self._bprimes, self._Cprimes, self._dprimes,
            self._Fprimes, self._gradFprimes, self._Gprimes,
            self_gradGprimes) = (Aprimes, bprimes, Cprimes, dprimes,
                    Fprimes, gradFprimes, Gprimes, gradGprimes)

        self.f = self.get_feasibility(Aprimes, bprimes, Cprimes, dprimes,
                Fprimes, Gprimes, self.eps)
        self.gradf = self.get_feasibility_grad(Aprimes, bprimes,
                Cprimes, dprimes, Fprimes, gradFprimes,
                Gprimes, gradGprimes, self.eps)
        self._solver = BoundedTraceSolver(self.f, self.gradf, self.dim+1)

    def transform_input(self, R, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
        """
        Transform input into correct form for feasibility solver.
        """
        m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
        Aprimes, Cprimes, Fprimes, gradFprimes, Gprimes, gradGprimes = \
                [], [], [], [], [], []
        dim = self.dim

        # Rescale the trace bound and expand all constraints to be
        # expressed in terms of Y
        for i in range(m):
            A = R * As[i]
            Aprime = np.zeros((dim+1,dim+1))
            Aprime[:dim, :dim] = A
            Aprimes.append(Aprime)
        bprimes = bs
        for j in range(n):
            C = R * Cs[j]
            Cprime = np.zeros((dim+1,dim+1))
            Cprime[:dim, :dim] = C
            Cprimes.append(Cprime)
        dprimes = ds
        for k in range(p):
            fk = Fs[k]
            gradfk = gradFs[k]
            def make_fprime(fk):
                return lambda Y: fk(R * Y[:dim,:dim])
            fprime = make_fprime(fk)
            Fprimes.append(fprime)
            def make_gradfprime(gradfk):
                def gradfprime(Y):
                    ret_grad = np.zeros((dim+1,dim+1))
                    ret_grad[:dim,:dim] = gradfk(R * Y[:dim,:dim])
                    return ret_grad
                return gradfprime
            gradfprime = make_gradfprime(gradfk)
            gradFprimes.append(gradfprime)
        for l in range(q):
            gl = Gs[l]
            gradgl = gradGs[l]
            def make_gprime(gl):
                return lambda Y: gl(R * Y[:dim,:dim])
            gprime = make_gprime(gl)
            Gprimes.append(gprime)
            def make_gradgprime(gradgl):
                def gradgprime(Y):
                    ret_grad = np.zeros((dim+1,dim+1))
                    ret_grad[:dim, :dim] = gradgl(R * Y[:dim,:dim])
                    return ret_grad
                return gradgprime
            gradgprime = make_gradgprime(gradgl)
            gradGprimes.append(gradgprime)
        return (Aprimes, bprimes, Cprimes, dprimes,
                    Fprimes, gradFprimes, Gprimes, gradGprimes)

    def get_feasibility(self, As, bs, Cs, ds, Fs, Gs, eps):
        m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
        M = compute_scale(m, n, p, q, eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        return f

    def get_feasibility_grad(self, As, bs, Cs, ds, Fs, gradFs, Gs,
            gradGs, eps):
        m, n, p, q = len(As), len(Cs), len(Fs), len(Gs)
        M = compute_scale(m, n, p, q, eps)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
        return gradf

    def feasibility_solve(self, N_iter, tol, X_init=None,
            methods=['frank_wolfe'], early_exit=True, disp=True,
            verbose=False, Rs=[10, 100], debug=False, min_step_size=1e-6,
            num_stable=np.inf):
        """
        Solves feasibility problems of the type

        Feasibility of X
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X)  = d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) <= R
        """
        for R in Rs:
            if debug:
                print "R: ", R
            self.init_solver(R)
            if X_init != None:
                dim = self.dim
                Y_init = np.zeros((dim+1, dim+1))
                Y_init[:dim, :dim] = X_init
                Y_init = Y_init / R
                init_trace = np.trace(Y_init)
                Y_init[dim, dim] = 1 - init_trace
                fY_init = self.f(Y_init)
            else:
                Y_init = None
            Y = self._solver.solve(N_iter, X_init=Y_init,
                    methods=methods, early_exit=early_exit, disp=verbose,
                    min_step_size=min_step_size, good_enough=-tol,
                    num_stable=num_stable)
            fY = self.f(Y)
            X, fX = R*Y[:self.dim, :self.dim], fY
            succeed = not (fX < -tol)
            if succeed:
                break
        return X, fX, succeed
