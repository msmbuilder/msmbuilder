from __future__ import print_function
import numpy as np
import scipy.linalg
from scipy.linalg import eigvals
from scipy.optimize import approx_fprime
from msmbuilder.cluster import NDGrid
from msmbuilder.example_datasets import load_doublewell
from msmbuilder.msm import MarkovStateModel, ContinuousTimeMSM
from msmbuilder.msm.core import _solve_msm_eigensystem


def test_0():
    # Verify that the partial derivatives of the ith eigenvalue of the
    # transition matrix with respect to the entries of the transition matrix
    # is given by the outer product of the left and right eigenvectors
    # corresponding to that eigenvalue.
    # \frac{\partial \lambda_k}{\partial T_{ij}} = U_{i,k} V_{j,k}

    X = load_doublewell(random_state=0)['trajectories']
    Y = NDGrid(n_bins_per_feature=10).fit_transform(X)
    model = MarkovStateModel(verbose=False).fit(Y)
    n = model.n_states_

    u, lv, rv = _solve_msm_eigensystem(model.transmat_, n)

    # first, compute forward difference numerical derivatives
    h = 1e-7
    dLambda_dP_numeric = np.zeros((n, n, n))
    # dLambda_dP_numeric[eigenvalue_index, i, j]
    for i in range(n):
        for j in range(n):
            # perturb the (i,j) entry of transmat
            H = np.zeros((n, n))
            H[i, j] = h
            u_perturbed = sorted(np.real(eigvals(model.transmat_ + H)), reverse=True)

            # compute the forward different approx. derivative of each
            # of the eigenvalues
            for k in range(n):
                # sort the eigenvalues of the perturbed matrix in descending
                # order, to be consistent w/ _solve_msm_eigensystem
                dLambda_dP_numeric[k, i, j] = (u_perturbed[k] - u[k]) / h

    for k in range(n):
        analytic = np.outer(lv[:, k], rv[:, k])
        np.testing.assert_almost_equal(dLambda_dP_numeric[k], analytic, decimal=5)


def test_1():
    X = load_doublewell(random_state=0)['trajectories']
    for i in range(10):
        Y = NDGrid(n_bins_per_feature=10).fit_transform([X[i]])
        model = MarkovStateModel(verbose=False).fit(Y)
        model2 = ContinuousTimeMSM().fit(Y)

        print(model.uncertainty_timescales())
        print(model2.uncertainty_timescales())
        print()

