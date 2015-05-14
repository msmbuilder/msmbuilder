import time
import numpy as np
from numpy import zeros
from numpy cimport npy_intp
from libc.math cimport exp, sqrt, fabs, log
from libc.stdlib cimport malloc, free


cdef class ReversibleHolmesRateMatrixEMFitter(object):
    """ReversibleHolmesRateMatrixEMFitter(countsmat, t)

    Holmes-Rubin Expectation-maximization algorithm for fitting
    reversible continous-time Markov models.

    This class implements "Algorithm 4" described by Metzner in [2], which was
    introduced by Holmes and Rubin in [1]. The notation follows [2].

    Parameters
    ----------
    countsmat : array, shape=(n, n)
        The matrix of observed transition counts.
    t : double
        The lag time.

    References
    ----------
    .. [1] Holmes, I., and G. M. Rubin. "An expectation maximization algorithm
        for training hidden substitution models." J. Mol. Biol. 317.5 (2002):
        753-764.
    .. [2] Metzner, Philipp, et al. "Generator estimation of Markov jump
        processes." J. Comp. Phys. 227.1 (2007): 353-375.
    """
    cdef double t
    cdef npy_intp n
    cdef double[:, ::1] countsmat

    def __cinit__(self, const double[:, ::1] countsmat, double t=1):
        if countsmat.shape[0] != countsmat.shape[0]:
            raise ValueError('countsmat must be square')

        self.countsmat = countsmat
        self.t = t
        self.n = countsmat.shape[0]

    def fit(self, const double[:, ::1] ratemat, int max_iter=100, double thresh=1e-6):
        """fit(ratemat, max_iter=100, thresh=1e-6)

        Iterate the EM algorithm to improve a guess rate matrix.

        Parameters
        ----------
        ratemat : array, shape=(n, n)
            The initial rate matrix.
        max_iter : int, optional
            Maximum number of EM iteration to perform.
        thresh : double, optinal
            Terminate EM when the change in the log-likelihood is less than
            this threshold.

        Returns
        -------
        ratemat : array, shape=(n, n)
            The new rate matrix.
        loglikelihoods : array, shape=(n_iters,)
            The log likelihood of the iterated rate matrix, after each step
            of EM.
        """
        cdef npy_intp i
        cdef double logl, iteration_start_time
        cdef double[:, ::1] new_ratemat
        cdef double[::1] loglikelihoods = zeros(max_iter)
        cdef double[::1] times = zeros(max_iter)

        if not (ratemat.shape[0] == ratemat.shape[0] == self.n):
            raise ValueError('ratemat must be (%d, %d)' % (self.n, self.n))

        for i in range(max_iter):
            iteration_start_time = time.time()
            new_ratemat, logl = self.em_step(ratemat)
            times[i] = time.time() - iteration_start_time
            loglikelihoods[i] = logl
            if i >= 1 and abs(logl-loglikelihoods[i-1]) < thresh:
                break
            ratemat = new_ratemat

        return np.asarray(ratemat), np.asarray(loglikelihoods[:i]), np.asarray(times[:i])

    def em_step(self, const double[:, ::1] ratemat):
        """Perform a single step of E-M

        Parameters
        ----------
        ratemat : array, shape=(n, n)
            The initial rate matrix.

        Returns
        -------
        new_ratemat : array, shape=(n, n)
            The updated rate matrix
        logl : double
            The log likelihood of new_ratemat
        """
        logl, pi, ER_i, EN_ij = self.e_step(ratemat)
        new_ratemat = self.m_step(pi, ER_i, EN_ij)
        return new_ratemat, logl

    def e_step(self, const double[:, ::1] ratemat):
        """e_step(ratemat)

        Perform a single E-step.

        Parameters
        ----------
        ratemat : array, shape=(n, n)
            The rate matrix.

        Returns
        -------
        logl : double
            The log likelihood of ratemat.
        pi : array, shape=(n,)
            The stationary distribution of the rate matrix.
        ER_i : array, shape=(n,),
            The expectation value (conditional on ratemat) of the amount of
            time the process has spent in each state. This is defined in Eq.
            19 of Metzner.
        EN_ij : array, shape=(n, n)
            The expectation value (conditional on ratemat) of number of
            transitions between each state. This is defined in Eq. 19 of
            Metzner.
        """
        cdef npy_intp i, c, d, p
        cdef double pi_norm
        cdef double[::1] w, pi, expwt
        cdef double[:, ::1] U, V, temp, tmat

        pi = zeros(self.n)
        expwt = zeros(self.n)
        temp = zeros((self.n, self.n))
        tmat = zeros((self.n, self.n))

        w, U, V = eig_K(ratemat, self.n, which='K')

        for i in range(self.n):
            pi[i] = U[i, 0]
            pi_norm += pi[i]
        for i in range(self.n):
            pi[i] /= pi_norm

        for i in range(self.n):
            expwt[i] = exp(w[i]*self.t)
        transmat(expwt, U, V, self.n, temp, tmat)

        cdef double[:, ::1] psi = self._build_psi(w)
        cdef double[:, :, :, ::1] p_kijl = self._build_p_abcd(U, V.T, psi)
        cdef double[:, :, ::1] R_ikl = self._build_R_ikl(tmat, p_kijl)
        cdef double[:, :, :, ::1] N_ijkl = self._build_N_ijkl(ratemat, tmat, p_kijl)
        cdef double[::1] ER_i = self._build_ER_i(R_ikl)
        cdef double[:, ::1] EN_ij = self._build_EN_ij(N_ijkl)
        cdef double logl = self._compute_logl(tmat)
        return logl, pi, ER_i, EN_ij

    def m_step(self, const double[::1] pi, const double[::1] ER_i, const double[:, ::1] EN_ij):
        """Perform a single M-step

        Parameters
        ----------
        pi : array, shape=(n,)
            The stationary distribution of the rate matrix.
        ER_i : array, shape=(n,),
            The expectation value (conditional on ratemat) of the amount of
            time the process has spent in each state. This is defined in Eq.
            19 of Metzner.
        EN_ij : array, shape=(n, n)
            The expectation value (conditional on ratemat) of number of
            transitions between each state. This is defined in Eq. 19 of
            Metzner.

        Returns
        -------
        new_ratemat : array, shape=(n, n)
            The new rate matrix.
        """

        cdef npy_intp i, j
        cdef double[:, ::1] ratemat = zeros((self.n, self.n))
        cdef double[:, ::1] mu = self._build_lagrange(pi, ER_i, EN_ij)

        for i in range(self.n):
            for j in range(self.n):
                if j != i:
                    ratemat[i, j] = EN_ij[i, j] / (-mu[i,j]*pi[i] + ER_i[i])
                    if ratemat[i, j] < 1e-15:
                        ratemat[i, j] = 0.0
                    ratemat[i, i] -= ratemat[i, j]
        return ratemat

    def _build_psi(self, const double[::1] w):
        # Build the matrix Psi_{pq}, defined in equation 27 of Metzner
        cdef double[:, ::1] psi = zeros((self.n, self.n))
        cdef npy_intp i, j
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    psi[i, j] = exprel(self.t*(w[i]-w[j])) * self.t * exp(w[j]*self.t)
                else:
                    psi[i, j] = self.t * exp(w[i]*self.t)
        return psi

    def _build_p_abcd(self, const double[:, ::1] U, const double[::1, :] VT, const double[:, ::1] psi):
        # Build the integral table p_{abcd} = \int_0^t ds p_{ab}(s) p_{cd}(t-s),
        # defined in equation 28 of Metzner.
        # In equation 24, the index labels are switched to 'k','i','j','l'.
        cdef npy_intp i, a, b, c, d, p, q
        cdef double[::1] rootpi = zeros(self.n)
        cdef double[::1] invrootpi = zeros(self.n)
        cdef double[:, :, ::1] U_V_PSI_cdp = zeros((self.n, self.n, self.n))
        cdef double[:, :, :, ::1] p_abcd = zeros((self.n, self.n, self.n, self.n))

        for c in range(self.n):
            for d in range(self.n):
                for p in range(self.n):
                    for q in range(self.n):
                        U_V_PSI_cdp[c,d,p] += U[c,q] * VT[q,d] * psi[p,q]

        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    for d in range(self.n):
                        for p in range(self.n):
                            p_abcd[a,b,c,d] += U[a,p] * VT[p,b] * U_V_PSI_cdp[c,d,p]

        return p_abcd

    def _build_R_ikl(self, const double[:, ::1] transmat, const double[:, :, :, ::1] p_kijl):
        # Build the R_{ikl} tensor defined in the first part of equation 24 of
        # Metzner.
        # R_{ikl} = E_L(R_i(t) | X(t) = l, X(0) = k)
        #         = \frac{1}{p_{kl}} \int_0^t ds p_{ki}(s) p_{il}(t-s),

        cdef npy_intp i, k, l
        cdef double[:, :, ::1] R_ikl = zeros((self.n, self.n, self.n))
        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.n):
                    R_ikl[i,k,l] = p_kijl[k,i,i,l]  / transmat[k,l]
        return R_ikl

    def _build_N_ijkl(self, const double[:, ::1] ratemat, const double[:, ::1] transmat,  const double[:, :, :, ::1] p_kijl):
        # Build the N_{ijkl} tensor defined in the second part of equation 24 of
        # Metzner.
        # N_{ijkl} = E_L(N_{ij}(t) | X(t) = l, X(0) = k)
        #          = \frac{l_{ij}}{p_{kl}} \int_0^t ds p_{ki}(s) p_{jl}(t-s),
        cdef npy_intp i, j, k, l
        cdef double[:, :, :, ::1] N_ijkl = zeros((self.n, self.n, self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        N_ijkl[i,j,k,l] = p_kijl[k, i, j, l] * ratemat[i, j] / transmat[k, l]
        return N_ijkl

    def _build_ER_i(self, const double[:, :, ::1] R_ikl):
        # Build the expectation value of the residence times, defined in
        # equation 19 (part 1) of Metzner.
        # ER_{i} = E_L(R_i(t | Y)) = \sum_{kl} C_{kl} R_{ikl}
        cdef npy_intp i, k, l
        cdef double[::1] ER_i = zeros(self.n)
        cdef double[:, ::1] countsmat = self.countsmat

        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.n):
                    ER_i[i] += countsmat[k, l] * R_ikl[i, k, l]
        return ER_i

    def _build_EN_ij(self, const double[:, :, :, ::1] N_ijkl):
        # Build the expectation value of the transition counts, defined in
        # equation 19 (part 2) of Metzner.
        # EN_{ij} = E_L(N_ij(t | Y)) = \sum_{kl} C_{kl} N_{ijkl}

        cdef npy_intp i, j, k, l
        cdef double[:, ::1] EN_ij = zeros((self.n, self.n))
        cdef double[:, ::1] countsmat = self.countsmat

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        EN_ij[i,j] += countsmat[k, l] * N_ijkl[i, j, k, l]
        return EN_ij

    def _build_lagrange(self, const double[::1] pi, const double[::1] ER_i, const double[:, ::1] EN_ij):
        # Build the lagrange multipliers for enforcing reversibility, \mu_{ij}
        # defined equation 30 of Metzner.
        cdef npy_intp i, j
        cdef double term1, term2, term3, term4
        cdef double[:, ::1] mu = zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                term1 = ER_i[j] / (pi[j] * EN_ij[j,i])
                term2 = ER_i[i] / (pi[i] * EN_ij[i,j])
                term3 = EN_ij[i,j] * EN_ij[j,i]
                term4 = EN_ij[i,j] + EN_ij[j,i]

                if fabs(EN_ij[i,j]) < 1e-16 and fabs(EN_ij[j,i]) < 1e-16:
                    mu[i,j] = 0
                elif term3 == 0:
                    mu[i,j] = 0
                else:
                    mu[i,j] = (term1 - term2) * (-term3 / term4)

        return mu

    def _compute_logl(self, const double[:, ::1] tmat):
        # Compute the log likelihood.
        cdef npy_intp i, j
        cdef double logl, logt
        cdef double[:, ::1] countsmat = self.countsmat

        logl = 0
        for i in range(self.n):
            for j in range(self.n):
                logt = log(tmat[i, j]) if tmat[i, j] > 1e-16 else 0
                logl += countsmat[i, j] * logt
        return logl
