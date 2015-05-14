import numpy as np
import scipy.linalg
from msmbuilder.msm._ratematrix import eig_K
from msmbuilder.msm._ratematrix import ReversibleHolmesRateMatrixEMFitter
from msmbuilder.msm._markovstatemodel import _transmat_mle_prinz


countsmat1 = np.array(
      [[  7.338e+03,   0.000e+00,   3.040e+02,   1.100e+01,   1.700e+01,
          2.961e+03,   1.300e+01,   2.497e+03],
       [  1.000e+00,   6.399e+03,   0.000e+00,   3.920e+02,   3.383e+03,
          9.100e+01,   2.476e+03,   3.000e+00],
       [  3.100e+02,   0.000e+00,   1.441e+04,   5.900e+01,   4.000e+00,
          2.200e+01,   0.000e+00,   3.482e+03],
       [  9.000e+00,   3.790e+02,   5.500e+01,   4.838e+03,   2.207e+03,
          2.400e+01,   3.270e+02,   6.500e+01],
       [  2.400e+01,   3.377e+03,   4.000e+00,   2.145e+03,   6.182e+03,
          7.800e+01,   3.408e+03,   3.900e+01],
       [  2.972e+03,   8.600e+01,   3.100e+01,   1.400e+01,   6.800e+01,
          4.181e+03,   6.600e+01,   6.170e+02],
       [  1.500e+01,   2.495e+03,   0.000e+00,   3.590e+02,   3.354e+03,
          5.900e+01,   6.869e+03,   1.300e+01],
       [  2.475e+03,   9.000e+00,   3.474e+03,   8.700e+01,   4.300e+01,
          6.210e+02,   6.000e+00,   4.742e+03]])


def _guess_ratemat():
    transmat, pi = _transmat_mle_prinz(countsmat1)
    ratemat = transmat - np.eye(transmat.shape[0])
    return ratemat


def test_1():
    ratemat = _guess_ratemat()
    transmat = np.ascontiguousarray(scipy.linalg.expm(ratemat))
    n = ratemat.shape[0]

    fitter = ReversibleHolmesRateMatrixEMFitter(countsmat1, t=1)

    w, U, V = eig_K(ratemat, n, which='K')
    psi = fitter._build_psi(w)
    p_kijl = np.ascontiguousarray(fitter._build_p_abcd(U, V.T, psi))
    p_kijl2 = np.einsum('ap,pb,cq,qd,pq->abcd', U, V.T, U, V.T, psi)

    r_ikl = np.array([p_kijl[:, i, i, :] / transmat for i in range(n)])
    n_ijkl = np.array([[ratemat[i,j] * p_kijl[:, i, j, :] / transmat for j in range(n)] for i in range(n)])
    ER_i = np.einsum('ikl,kl->i', r_ikl, countsmat1)
    EN_ij = np.einsum('ijkl,kl', n_ijkl, countsmat1)

    np.testing.assert_array_almost_equal(
        p_kijl, p_kijl2)
    np.testing.assert_array_almost_equal(
        r_ikl, fitter._build_R_ikl(transmat, p_kijl))
    np.testing.assert_array_almost_equal(
        ER_i, fitter._build_ER_i(r_ikl))
    np.testing.assert_array_almost_equal(
        EN_ij, fitter._build_EN_ij(n_ijkl))


def test_2():
    ratemat = _guess_ratemat()
    transmat = np.ascontiguousarray(scipy.linalg.expm(ratemat))
    n = ratemat.shape[0]

    fitter = ReversibleHolmesRateMatrixEMFitter(countsmat1, t=1)
    ratemat, logl, times = fitter.fit(ratemat)

    pi_norm = 0
    pi = np.zeros(n)
    w, U, V = eig_K(ratemat, n, which='K')
    for i in range(n):
        pi[i] = U[i, 0]
        pi_norm += pi[i]
    for i in range(n):
        pi[i] /= pi_norm

    for i in range(n):
        for j in range(n):
            assert abs(pi[i] * ratemat[i,j] - pi[j] * ratemat[j,i]) < 1e-12

    print(ratemat)
    # print(logl)
    # print(times)
    # print(np.asarray(logl))
    # print(np.diff(logl))
    # import matplotlib.pyplot as plt
    # plt.plot(logl)
    # plt.show()
