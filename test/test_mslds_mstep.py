import numpy as np
import warnings
import mdtraj as md
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mslds_examples import AlanineDipeptideModel
from mixtape.mslds_solver import MetastableSwitchingLDSSolver
from mixtape.mslds_solver import AQb_solve, A_solve, Q_solve
from sklearn.hmm import GaussianHMM
from test_mslds_estep import reference_estep
from mixtape.datasets.alanine_dipeptide import fetch_alanine_dipeptide
from mixtape.datasets.alanine_dipeptide import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_ALANINE
from mixtape.datasets.met_enkephalin import fetch_met_enkephalin
from mixtape.datasets.met_enkephalin import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_MET
from mixtape.datasets.base import get_data_home
from os.path import join

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
    AQb_solve(dim, A, Q, mu, B, C, D, E, F)

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
    AQb_solve(dim, A, Q, mu, B, C, D, E, F)

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
    A_solve(block_dim, B, C, D, E, Q, mu)

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
    A_solve(block_dim, B, C, D, E, Q, mu)

def test_Q_solve_muller():
    block_dim = 2
    np.set_printoptions(precision=4)

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
    Q = Q_solve(block_dim, A, D, F, disp=True, debug=False,
            verbose=False, Rs=[100])
    print "D:\n", D
    assert Q != None
    assert np.linalg.norm(Q, 2) < np.linalg.norm(D, 2)

def test_A_solve_muller():
    block_dim = 2
    B = np.array([[208.27749525,  -597.11827148],
                   [ -612.99179464, 1771.25551671]])

    C = np.array([[202.83070879, -600.32796941],
                   [-601.76432584, 1781.07130791]])

    D = np.array([[0.00326556, 0.00196009],
                   [0.00196009, 0.00322879]])

    E = np.array([[205.80695137, -599.79918374],
                  [-599.79918374, 1782.52514543]])
    Q = .9 * D
    mu =  np.array([[-0.7010104, 1.29133034]])
    mu = np.reshape(mu, (block_dim, 1))
    A_solve(block_dim, B, C, D, E, Q, mu, verbose=False, disp=True)

def test_Q_solve_muller_2():
    block_dim = 2.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.set_printoptions(precision=4)
    B = np.array([[  359.92406863,  -853.5934402 ],
                  [ -842.86780552,  2010.34907067]])

    C = np.array([[  361.80793384,  -850.60352492],
                  [ -851.82693628,  2002.62881727]])

    D = np.array([[ 0.00261965,  0.00152437],
                  [ 0.00152437,  0.00291518]])

    E = np.array([[  364.88271615,  -849.83206073],
                  [ -849.83206073,  2004.72145185]])

    F = np.array([[ 2.72226628,  1.60237858],
                  [ 1.60237858,  3.0191094 ]])
    A = np.zeros((block_dim, block_dim))
    Q = Q_solve(block_dim, A, D, F, disp=True,
            verbose=False, Rs=[100])
    print "D:\n", D
    assert Q != None
    assert np.linalg.norm(Q, 2) < np.linalg.norm(D, 2)

def test_Q_solve_muller_3():
    block_dim = 2.
    from numpy import array, float32
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.set_printoptions(precision=4)
    A = (array([[ 0.80000001, -0.2       ],
                [-0.2       ,  0.80000001]], dtype=float32))
    A = np.zeros((block_dim, block_dim))
    D = (array([[ 0.00515675,  0.00027678],
                [ 0.00027678,  0.0092519 ]], dtype=float32))
    F = (array([[ 5.79813337, -2.13557243],
                [-2.13554192, -6.50420761]], dtype=float32))
    Q = Q_solve(block_dim, A, D, F, disp=True,
            verbose=False, Rs=[100])
    print "D:\n", D
    assert Q != None
    assert np.linalg.norm(Q, 2) < np.linalg.norm(D, 2)


def test_A_solve_muller_2():
    block_dim = 2.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.set_printoptions(precision=4)
    B = np.array([[  359.92406863,  -853.5934402 ],
                  [ -842.86780552,  2010.34907067]])

    C = np.array([[  361.80793384,  -850.60352492],
                  [ -851.82693628,  2002.62881727]])

    D = np.array([[ 0.00261965,  0.00152437],
                  [ 0.00152437,  0.00291518]])

    E = np.array([[  364.88271615,  -849.83206073],
                  [ -849.83206073,  2004.72145185]])
    Q = .9 * D
    Dinv = np.linalg.inv(D)
    Qinv = np.linalg.inv(Q)
    mu =  np.array([[-0.7010104, 1.29133034]])
    mu = np.reshape(mu, (block_dim, 1))
    A_solve(block_dim, B, C, D, E, Q, mu, verbose=False, disp=True)

def test_A_solve_muller_3():
    block_dim = 2
    Q = np.array([[ 0.00268512, -0.00030655],
                 [-0.00030655,  0.002112  ]])
    mu = np.array([[ 0.58044142,  0.03486499]])
    mu = np.reshape(mu, (block_dim, 1))
    B = np.array([[ 269.81124024,   15.28704689],
                    [  16.32464053,    0.99806799]])

    C = np.array([[ 266.55743817,   16.13788517],
                    [  16.01112828,    0.96934361]])

    D = np.array([[ 0.00246003, -0.00017837],
                    [-0.00017837,  0.00190514]])

    E = np.array([[ 267.86405002,   15.94161187],
                    [  15.94161187,    2.47997446]])

    F = np.array([[ 1.97090458, -0.15635765],
                    [-0.15635765,  1.50541836]])
    A_solve(block_dim, B, C, D, E, Q, mu, verbose=False, disp=True)

def test_A_solve_muller_4():
    block_dim = 2
    B = (np.array([[  47.35392822,  -87.69193367],
                [ -83.75658227,  155.95421092]]))
    C = (np.array([[  47.72109794,  -84.70032366],
                [ -86.21727938,  153.02731464]]))
    D = (np.array([[ 0.34993938, -0.3077952 ],
                [-0.3077952 ,  0.854263  ]]))
    E = (np.array([[  48.90734354, -86.23552793],
                [ -86.23552793, 153.58913243]]))
    Q = (np.array([[ 0.00712922, -0.00164496],
                [-0.00164496, 0.01020176]]))
    mu = (np.array([-0.70567481, 1.27493635]))
    A_solve(block_dim, B, C, D, E, Q, mu, verbose=False, disp=True)

def test_Q_solve_plusmin():
    block_dim = 1
    A = np.array([[.0]])
    D = np.array([[.0204]])
    F = np.array([[25.47]])
    Q = Q_solve(block_dim, A, D, F)
    print "D:\n", D
    assert Q != None
    assert np.linalg.norm(Q, 2) < np.linalg.norm(D, 2)

def test_plusmin_mstep():
    # Set constants
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
    import pdb, traceback, sys
    try:
        # Set constants
        n_seq = 1
        num_trajs = 1
        T = 2500

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
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def A_solve_test_alanine():
	#Auto-generated test case from failing run of
	#A-solve:
	import numpy as np
	import pickle
	from mixtape.mslds_solver import AQb_solve, A_solve, Q_solve
	block_dim = 66
	B = pickle.load(open("B_alanine_1.p", "r"))
	C = pickle.load(open("C_alanine_1.p", "r"))
	D = pickle.load(open("D_alanine_1.p", "r"))
	E = pickle.load(open("E_alanine_1.p", "r"))
	Q = pickle.load(open("Q_alanine_1.p", "r"))
	mu = pickle.load(open("mu_alanine_1.p", "r"))
	A_solve(block_dim, B, C, D, E, Q, mu,
		disp=True, debug=False, verbose=True,
		Rs=[100], N_iter=150)

def test_alanine_dipeptide_mstep():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        b = fetch_alanine_dipeptide()
        trajs = b.trajectories
        # While debugging, restrict to first trajectory only
        trajs = [trajs[0]]
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_ALANINE)
        top = md.load(join(data_dir, 'ala2.pdb'))
        n_components = 2
        # Superpose m
        data = []
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (n_frames,n_features), order='F')
            data.append(Z)

        # Fit reference model and initial MSLDS model
        print "Starting Gaussian Model Fit"
        refmodel = GaussianHMM(n_components=n_components,
                            covariance_type='full').fit(data)
        print "Done with Gaussian Model Fit"

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
        solver.do_mstep(As, Qs, bs, means, covars, rstats, N_iter=100)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_met_enkephalin_mstep():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        b = fetch_met_enkephalin()
        trajs = b.trajectories
        # While debugging, restrict to first trajectory only
        trajs = [trajs[0]]
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        print "n_features: ", n_features

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_MET)
        top = md.load(join(data_dir, '1plx.pdb'))
        n_components = 2

        # Superpose m
        data = []
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (n_frames, n_features), order='F')
            data.append(Z)

        # Fit reference model and initial MSLDS model
        print "Starting Gaussian Model Fit"
        refmodel = GaussianHMM(n_components=n_components,
                            covariance_type='full').fit(data)
        print "Done with Gaussian Model Fit"

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
        solver.do_mstep(As, Qs, bs, means, covars, rstats, N_iter=100,
                            verbose=True)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
