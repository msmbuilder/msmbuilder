import numpy as np
import warnings
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mixtape.mslds_solver import MetastableSwitchingLDSSolver
from mixtape.mslds_solver import AQb_solve, A_solve, Q_solve
from sklearn.hmm import GaussianHMM
from test_mslds_estep import reference_estep

def test_AQb_solve_simple():
    dim = 1
    A = np.array([[.5]])
    Q = np.array([[.1]])
    Qinv = np.array([[10.]])
    mu = np.array([[0.]])
    B = np.array([[1.]])
    C = np.array([[2.]])
    D = np.array([[1.]])
    Dinv = np.array([[1.]])
    E = np.array([[1.]])
    F = np.array([[1.]])
    AQb_solve(dim, A, Q, Qinv, mu, B, C, D, Dinv, E, F)

def test_AQb_solve_plusmin():
    # Numbers below were generated from a sample run of
    # plusmin
    dim = 1
    A = np.array([[.0]])
    Q = np.array([[.02]])
    Qinv = np.array([[48.99]])
    mu = np.array([[.991]])
    B = np.array([[1238.916]])
    C = np.array([[1225.025]])
    D = np.array([[.0204]])
    Dinv = np.array([[49.02]])
    E = np.array([[48.99]])
    F = np.array([[25.47]])
    AQb_solve(dim, A, Q, Qinv, mu, B, C, D, Dinv, E, F)

def test_A_solve_plusmin():
    block_dim = 1
    B = np.array([[1238.916]])
    C = np.array([[1225.025]])
    D = np.array([[.0204]])
    Dinv = np.array([[49.02]])
    E = np.array([[48.99]])
    Q = np.array([[.02]])
    Qinv = np.array([[48.99]])
    mu = np.array([[1.]])
    A_solve(block_dim, B, C, D, Dinv, E, Q, Qinv, mu)

def test_Q_solve_plusmin():
    block_dim = 1
    A = np.array([[.0]])
    D = np.array([[.0204]])
    Dinv = np.array([[49.02]])
    F = np.array([[25.47]])
    Q_solve(block_dim, A, D, Dinv, F)

def test_plusmin_mstep():
    # Set constants
    num_hotstart = 3
    n_seq = 1
    T = 2000

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = plusmin.x_dim
    n_components = plusmin.K

    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Obtain sufficient statistics from refmodel
    rlogprob, rstats = reference_estep(refmodel, data)
    means = refmodel.means_
    covars = refmodel.covars_
    transmat = refmodel.transmat_
    populations = refmodel.startprob_
    As = []
    for i in range(n_components):
        As.append(np.zeros((n_features, n_features)))
    Qs = refmodel.covars_
    bs = refmodel.means_
    means = refmodel.means_
    covars = refmodel.covars_

    # Test AQB solver for MSLDS
    solver = MetastableSwitchingLDSSolver(n_components, n_features)
    solver.do_mstep(As, Qs, bs, means, covars, rstats)

def test_muller_potential_mstep():
    # Set constants
    n_seq = 1
    num_trajs = 1
    T = 2500
    num_hotstart = 0

    # Generate data
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    muller = MullerModel()
    data, trajectory, start = \
            muller.generate_dataset(n_seq, num_trajs, T)
    n_features = muller.x_dim
    n_components = muller.K

    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)

    # Obtain sufficient statistics from refmodel
    rlogprob, rstats = reference_estep(refmodel, data)
    means = refmodel.means_
    covars = refmodel.covars_
    transmat = refmodel.transmat_
    populations = refmodel.startprob_
    As = []
    for i in range(n_components):
        As.append(np.zeros((n_features, n_features)))
    Qs = refmodel.covars_
    bs = refmodel.means_
    means = refmodel.means_
    covars = refmodel.covars_

    # Test AQB solver for MSLDS
    solver = MetastableSwitchingLDSSolver(n_components, n_features)
    solver.do_mstep(As, Qs, bs, means, covars, rstats)
