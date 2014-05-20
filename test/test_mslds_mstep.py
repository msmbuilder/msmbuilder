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
    E = np.array([[48.99]])
    Q = np.array([[.02]])
    Dinv = np.array([[49.02]])
    Qinv = np.array([[48.99]])
    mu = np.array([[1.]])
    A_solve(block_dim, B, C, D, Dinv, E, Q, Qinv, mu)

def test_A_solve_plusmin_2():
    block_dim = 1
    B = np.array([[ 965.82431552]])
    C = np.array([[ 950.23843989]])
    D = np.array([[ 0.02430409]])
    E = np.array([[ 974.49540394]])
    F = np.array([[ 24.31867657]])
    Q = np.array([[ 0.02596519]])
    mu = np.array([[ -1.]])
    Dinv = np.linalg.inv(D)
    Qinv = np.linalg.inv(Q)
    A_solve(block_dim, B, C, D, Dinv, E, Q, Qinv, mu)

def test_Q_solve_muller():
    block_dim = 2
    np.set_printoptions(precision=2)

    A = np.zeros((block_dim, block_dim))
    B = np.array([[208.27749525,  -597.11827148],
                   [ -612.99179464, 1771.25551671]])

    C = np.array([[202.83070879, -600.32796941],
                   [-601.76432584, 1781.07130791]])

    D = np.array([[0.00326556, 0.00196009],
                   [0.00196009, 0.00322879]])

    E = np.array([[205.80695137, -599.79918374],
                  [-599.79918374, 1782.52514543]])

    F = np.array([[2.62197238, 1.58163533],
                  [1.58163533, 2.58977211]])
    Dinv = np.linalg.inv(D)
    Q_solve(block_dim, A, D, Dinv, F, disp=True, debug=True,
            verbose=True, Rs=[100])



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
