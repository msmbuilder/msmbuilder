import string
import numpy as np
import warnings
import mdtraj as md
import os
from sklearn.hmm import GaussianHMM
from sklearn.utils.extmath import logsumexp
from mixtape.mslds import MetastableSwitchingLDS
from mixtape import _mslds
from mixtape.ghmm import GaussianFusionHMM
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mslds_examples import AlanineDipeptideModel
import matplotlib.pyplot as plt
from mixtape.datasets.alanine_dipeptide import fetch_alanine_dipeptide
from mixtape.datasets.alanine_dipeptide import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_ALANINE
from mixtape.datasets.base import get_data_home
from os.path import join
from nose.plugins.skip import Skip, SkipTest
from nose.plugins.attrib import attr


@attr('broken')
def reference_estep(refmodel, data):
    curr_logprob = 0
    stats = refmodel._initialize_sufficient_statistics()
    stats['post[1:]'] = np.zeros(refmodel.n_components)
    stats['post[:-1]'] = np.zeros(refmodel.n_components)
    stats['obs[1:]'] = np.zeros((refmodel.n_components,
                                    refmodel.n_features))
    stats['obs[:-1]'] = np.zeros((refmodel.n_components,
                                    refmodel.n_features))
    stats['obs*obs[t-1].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))
    stats['obs[1:]*obs[1:].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))
    stats['obs[:-1]*obs[:-1].T'] = np.zeros((refmodel.n_components,
                            refmodel.n_features, refmodel.n_features))

    for seq in data:
        framelogprob = refmodel._compute_log_likelihood(seq)
        lpr, fwdlattice = refmodel._do_forward_pass(framelogprob)
        bwdlattice = refmodel._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        curr_logprob += lpr
        refmodel._accumulate_sufficient_statistics(stats, seq,
                            framelogprob, posteriors, fwdlattice,
                            bwdlattice, string.ascii_letters)

        # accumulate the extra stats that our model has but the
        # sklearn model doesn't
        stats['post[1:]'] += posteriors[1:].sum(axis=0)
        stats['post[:-1]'] += posteriors[:-1].sum(axis=0)
        stats['obs[1:]'] += np.dot(posteriors[1:].T, seq[1:])
        stats['obs[:-1]'] += np.dot(posteriors[:-1].T, seq[:-1])

        for t in range(1, len(seq)):
            obsobsTminus1 = np.outer(seq[t], seq[t-1])
            for c in range(refmodel.n_components):
                stats['obs*obs[t-1].T'][c] += \
                        posteriors[t, c] * obsobsTminus1

        for t in range(1, len(seq)):
            obsobsT = np.outer(seq[t], seq[t])
            for c in range(refmodel.n_components):
                stats['obs[1:]*obs[1:].T'][c] += \
                        posteriors[t, c] * obsobsT

        for t in range(len(seq)-1):
            obsobsT = np.outer(seq[t], seq[t])
            for c in range(refmodel.n_components):
                stats['obs[:-1]*obs[:-1].T'][c] += \
                        posteriors[t, c] * obsobsT
    return curr_logprob, stats


@attr('broken')
def test_plusmin_stats():
    # Set constants
    num_hotstart = 3
    n_seq = 1
    T = 2000

    # Generate data
    plusmin = PlusminModel()
    data, hidden = plusmin.generate_dataset(n_seq, T)
    n_features = plusmin.x_dim
    n_components = plusmin.K

    # Fit reference model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Fit initial MSLDS model from reference model
    model = MetastableSwitchingLDS(n_components, n_features,
                                n_hotstart=0)
    model.inferrer._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    model.As_ = [np.zeros((n_features, n_features)),
                    np.zeros((n_features, n_features))]
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model.inferrer.do_estep()
    rlogprob, rstats = reference_estep(refmodel, data)

    yield lambda: np.testing.assert_array_almost_equal(stats['post'],
            rstats['post'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'],
            rstats['post[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'],
            rstats['post[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'],
            rstats['obs'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'],
            rstats['obs[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'],
            rstats['obs[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'],
            rstats['obs*obs.T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['trans'], rstats['trans'], decimal=1)


@attr('broken')
def test_muller_potential_stats():
    raise SkipTest('Not ready yet')

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
    model = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=num_hotstart)
    model.inferrer._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    As = []
    for i in range(n_components):
        As.append(np.zeros((n_features, n_features)))
    model.As_ = As
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model.inferrer.do_estep()
    rlogprob, rstats = reference_estep(refmodel, data)

    yield lambda: np.testing.assert_array_almost_equal(stats['post'],
            rstats['post'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'],
            rstats['post[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'],
            rstats['post[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'],
            rstats['obs'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'],
            rstats['obs[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'],
            rstats['obs[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'],
            rstats['obs*obs.T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['trans'], rstats['trans'], decimal=1)


@attr('broken')
def test_alanine_dipeptide_stats():
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
        Z = np.reshape(Z, (n_frames, n_features), order='F')
        data.append(Z)

    n_hotstart = 3
    # Fit reference model and initial MSLDS model
    refmodel = GaussianHMM(n_components=n_components,
                        covariance_type='full').fit(data)
    rlogprob, rstats = reference_estep(refmodel, data)

    model = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=n_hotstart)
    model.inferrer._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    As = []
    for i in range(n_components):
        As.append(np.zeros((n_features, n_features)))
    model.As_ = As
    Qs = []
    eps = 1e-7
    for i in range(n_components):
        Q = refmodel.covars_[i] + eps*np.eye(n_features)
        Qs.append(Q)
    model.Qs_ = Qs
    model.bs_ = refmodel.means_
    logprob, stats = model.inferrer.do_estep()

    yield lambda: np.testing.assert_array_almost_equal(stats['post'],
            rstats['post'], decimal=2)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'],
            rstats['post[1:]'], decimal=2)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'],
            rstats['post[:-1]'], decimal=2)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'],
            rstats['obs'], decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'],
            rstats['obs[1:]'], decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'],
            rstats['obs[:-1]'], decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'],
            rstats['obs*obs.T'], decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'],
            decimal=1)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'],
            decimal=1)
    # This test fails consistently. TODO: Figure out why.
    #yield lambda: np.testing.assert_array_almost_equal(
    #        stats['trans'], rstats['trans'], decimal=2)


@attr('broken')
def test_randn_stats():
    """
    Sanity test MSLDS sufficient statistic gathering by setting
    dynamics model to 0 and testing that E-step matches that of
    HMM
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Generate reference data
    n_states = 2
    n_features = 3
    data = [np.random.randn(100, n_features),
            np.random.randn(100, n_features)]
    refmodel = GaussianHMM(n_components=n_states,
                        covariance_type='full').fit(data)

    # test all of the sufficient statistics against sklearn and pure python

    model = MetastableSwitchingLDS(n_states=n_states,
            n_features=n_features, n_hotstart=0)
    model.inferrer._sequences = data
    model.means_ = refmodel.means_
    model.covars_ = refmodel.covars_
    model.transmat_ = refmodel.transmat_
    model.populations_ = refmodel.startprob_
    # Is there a more elegant way to do this?
    model.As_ = [np.zeros((n_features, n_features)),
                    np.zeros((n_features, n_features))]
    model.Qs_ = refmodel.covars_
    model.bs_ = refmodel.means_

    iteration = 0 # Remove this step once hot_start is factored out
    logprob, stats = model.inferrer.do_estep()
    rlogprob, rstats = reference_estep(refmodel, data)

    yield lambda: np.testing.assert_array_almost_equal(stats['post'],
            rstats['post'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[1:]'],
            rstats['post[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['post[:-1]'],
            rstats['post[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs'],
            rstats['obs'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[1:]'],
            rstats['obs[1:]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs[:-1]'],
            rstats['obs[:-1]'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(stats['obs*obs.T'],
            rstats['obs*obs.T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs*obs[t-1].T'], rstats['obs*obs[t-1].T'], decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[1:]*obs[1:].T'], rstats['obs[1:]*obs[1:].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['obs[:-1]*obs[:-1].T'], rstats['obs[:-1]*obs[:-1].T'],
            decimal=3)
    yield lambda: np.testing.assert_array_almost_equal(
            stats['trans'], rstats['trans'], decimal=3)


@attr('broken')
def test_gaussian_loglikelihood_full():
    _mslds.test_gaussian_loglikelihood_full()
