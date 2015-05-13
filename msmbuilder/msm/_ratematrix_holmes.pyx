import numpy as np
from numpy import zeros
from numpy cimport npy_intp
from libc.math cimport exp, sqrt, fabs, log
from libc.stdlib cimport malloc, free


cdef class ReversibleHolmesRateMatrixEMFitter(object):
    cdef double t
    cdef npy_intp n
    cdef double[:, ::1] countsmat

    def __cinit__(self, const double[:, ::1] countsmat, double t=1):
        if countsmat.shape[0] != countsmat.shape[0]:
            raise ValueError('countsmat must be square')

        self.countsmat = countsmat
        self.t = t
        self.n = countsmat.shape[0]

    def fit(self, const double[:, ::1] ratemat, max_iter=100, min_iter=10, thresh=1e-6):
        cdef npy_intp i
        cdef double logl
        cdef double[:, ::1] new_ratemat
        cdef double[::1] loglikelihoods = zeros(max_iter)

        for i in range(max_iter):
            new_ratemat, logl = self.em_step(ratemat)
            loglikelihoods[i] = logl
            if i >= min_iter and (logl-loglikelihoods[i-1]) < thresh:
                break
            ratemat = new_ratemat
        return ratemat, loglikelihoods[:i]

    def em_step(self, const double[:, ::1] ratemat):
        cdef double[::1] pi
        cdef double[::1] ER_i
        cdef double[:, ::1] EN_ij
        cdef double[:, ::1] new_ratemat

        logl, pi, ER_i, EN_ij = self.e_step(ratemat)
        new_ratemat = self.m_step(pi, ER_i, EN_ij)
        return new_ratemat, logl

    def e_step(self, const double[:, ::1] ratemat):
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
        cdef npy_intp i, j
        cdef double[:, ::1] ratemat = zeros((self.n, self.n))
        cdef double[:, ::1] mu = self._build_lagrange(pi, ER_i, EN_ij)

        for i in range(self.n):
            for j in range(self.n):
                if j != i:
                    ratemat[i, j] = EN_ij[i, j] / (-mu[i,j]*pi[i] + ER_i[i])
                    ratemat[i, i] -= ratemat[i, j]
        return ratemat

    def _build_psi(self, const double[::1] w):
        # equation 27
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
        # equation 28
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
        # equation 24, part 1

        cdef npy_intp i, k, l
        cdef double[:, :, ::1] R_ikl = zeros((self.n, self.n, self.n))
        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.n):
                    R_ikl[i,k,l] = p_kijl[k,i,i,l]  / transmat[k,l]
        return R_ikl

    def _build_N_ijkl(self, const double[:, ::1] ratemat, const double[:, ::1] transmat,  const double[:, :, :, ::1] p_kijl):
        # equation 24 part 2
        cdef npy_intp i, j, k, l
        cdef double[:, :, :, ::1] N_ijkl = zeros((self.n, self.n, self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        N_ijkl[i,j,k,l] = p_kijl[k, i, j, l] * ratemat[i, j] / transmat[k, l]
        return N_ijkl

    def _build_ER_i(self, const double[:, :, ::1] R_ikl):
        # equation 19, part 1
        cdef npy_intp i, k, l
        cdef double[::1] ER_i = zeros(self.n)

        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.n):
                    ER_i[i] += self.countsmat[k, l] * R_ikl[i, k, l]
        return ER_i

    def _build_EN_ij(self, const double[:, :, :, ::1] N_ijkl):
        # equation 19, part 2
        cdef npy_intp i, j, k, l
        cdef double[:, ::1] EN_ij = zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        EN_ij[i,j] += self.countsmat[k, l] * N_ijkl[i, j, k, l]
        return EN_ij

    def _build_lagrange(self, const double[::1] pi, const double[::1] ER_i, const double[:, ::1] EN_ij):
        # equation 30
        cdef npy_intp i, j
        cdef double[:, ::1] mu = zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                term1 = ER_i[j] / (pi[j] * EN_ij[j,i])
                term2 = ER_i[i] / (pi[i] * EN_ij[i,j])
                term3 = EN_ij[i,j] * EN_ij[j,i]
                term4 = EN_ij[i,j] + EN_ij[j,i]

                if fabs(EN_ij[i,j]) < 1e-16 and fabs(EN_ij[j,i]) < 1e-16:
                    mu[i,j] = 0
                else:
                    mu[i,j] = (term1 - term2) * (-term3 / term4)
        return mu

    def _compute_logl(self, const double[:, ::1] tmat):
        cdef npy_intp i, j
        cdef double logl, logt

        logl = 0
        for i in range(self.n):
            for j in range(self.n):
                logt = log(tmat[i, j]) if tmat[i, j] > 1e-16 else 0
                logl += self.countsmat[i, j] * logt
        return logl
