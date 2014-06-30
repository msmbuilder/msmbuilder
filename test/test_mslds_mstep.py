import numpy as np
import warnings
import mdtraj as md
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mslds_examples import AlanineDipeptideModel
from mixtape.mslds_solver import MetastableSwitchingLDSSolver
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
    n_components = 1
    n_features = 1
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
    g = 1.
    iteration = 0.
    solver = MetastableSwitchingLDSSolver(n_components, n_features)
    solver.AQb_solve(A, Q, mu, B, C, D, E, F, g, iteration)

def test_AQb_solve_plusmin():
    # Numbers below were generated from a sample run of
    # plusmin
    n_components = 1
    n_features = 1
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
    g = 1.
    iteration = 0.
    solver = MetastableSwitchingLDSSolver(n_components, n_features)
    solver.AQb_solve(A, Q, mu, B, C, D, E, F, g, iteration)

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

def test_plusmin_mstep_no_stability():
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
    solver.do_mstep(As, Qs, bs, means, covars, rstats, stable=False)

def test_muller_potential_mstep_no_stability():
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
        solver.do_mstep(As, Qs, bs, means, covars, rstats, stable=False)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


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
        tol=1e-1
        search_tol=1

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
        solver.do_mstep(As, Qs, bs, means, covars, rstats, tol=tol,
                        verbose=False, search_tol=search_tol)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_alanine_dipeptide_mstep_no_stability():
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
        tol=1e-1
        search_tol=1

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
        solver.do_mstep(As, Qs, bs, means, covars, rstats, tol=tol,
                        verbose=False, search_tol=search_tol, stable=False)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
