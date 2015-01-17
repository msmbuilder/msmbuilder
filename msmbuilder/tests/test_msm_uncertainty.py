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
    random = np.random.RandomState(0)
    h = 1e-7
    X = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 2,2,2,2,2,1, 1, 1, 1 ,1, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    model = MarkovStateModel().fit(X)
    n = model.n_states_

    u, lv, rv = _solve_msm_eigensystem(model.transmat_, n)

    for k in range(n):
        dLambda_dP_numeric = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H = np.zeros((n, n))
                H[i, j] = h
                w = -np.sort(-eigvals(model.transmat_ + H))
                dLambda_dP_numeric[i, j] = (w[k] - model.eigenvalues_[k]) / h


        dLambda_dP_numeric
        analytic = np.outer(lv[:, k], rv[:, k])
        np.testing.assert_almost_equal(dLambda_dP_numeric, analytic)

def test_1():
    X = load_doublewell(random_state=0)['trajectories']
    for i in range(10):
        Y = NDGrid(n_bins_per_feature=10).fit_transform([X[i]])
        model = MarkovStateModel(verbose=False).fit(Y)
        model2 = ContinuousTimeMSM().fit(Y)

        print(model.uncertainty_timescales())
        print(model2.uncertainty_timescales())
        print()

