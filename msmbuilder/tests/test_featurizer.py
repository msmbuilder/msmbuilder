import numpy as np
from mdtraj import compute_dihedrals, compute_phi
from mdtraj.testing import eq
from scipy.stats import vonmises as vm

from msmbuilder.example_datasets import AlanineDipeptide, MinimalFsPeptide
from msmbuilder.featurizer import get_atompair_indices, FunctionFeaturizer, \
    DihedralFeaturizer, AtomPairsFeaturizer, SuperposeFeaturizer, \
    RMSDFeaturizer, VonMisesFeaturizer, Slicer


def test_function_featurizer():
    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0]

    # use the dihedral to compute phi for ala
    atom_ind = [[4, 6, 8, 14]]
    func = compute_dihedrals
    # test with args
    f = FunctionFeaturizer(func, func_args={"indices": atom_ind})
    res1 = f.transform([trj0])

    # test with function in a function without any args
    def funcception(trj):
        return compute_phi(trj)[1]

    f = FunctionFeaturizer(funcception)
    res2 = f.transform([trj0])

    # know results
    f3 = DihedralFeaturizer(['phi'], sincos=False)
    res3 = f3.transform([trj0])

    # compare all
    for r in [res2, res3]:
        np.testing.assert_array_almost_equal(res1, r)


def test_that_all_featurizers_run():
    # TODO: include all featurizers, perhaps with generator tests

    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)

    featurizer = AtomPairsFeaturizer(pair_indices)
    X_all = featurizer.transform(trajectories)

    featurizer = SuperposeFeaturizer(np.arange(15), trj0)
    X_all = featurizer.transform(trajectories)

    featurizer = DihedralFeaturizer(["phi", "psi"])
    X_all = featurizer.transform(trajectories)

    featurizer = VonMisesFeaturizer(["phi", "psi"])
    X_all = featurizer.transform(trajectories)

    # Below doesn't work on ALA dipeptide
    # featurizer = msmbuilder.featurizer.ContactFeaturizer()
    # X_all = featurizer.transform(trajectories)

    featurizer = RMSDFeaturizer(trj0)
    X_all = featurizer.transform(trajectories)


def test_von_mises_featurizer():
    trajectories = AlanineDipeptide().get_cached().trajectories

    featurizer = VonMisesFeaturizer(["phi"], n_bins=18)
    X_all = featurizer.transform(trajectories)
    n_frames = trajectories[0].n_frames
    assert X_all[0].shape == (n_frames, 18), (
        "unexpected shape returned: (%s, %s)" %
        X_all[0].shape)

    featurizer = VonMisesFeaturizer(["phi", "psi"], n_bins=18)
    X_all = featurizer.transform(trajectories)
    n_frames = trajectories[0].n_frames
    assert X_all[0].shape == (n_frames, 36), (
        "unexpected shape returned: (%s, %s)" %
        X_all[0].shape)

    featurizer = VonMisesFeaturizer(["phi", "psi"], n_bins=10)
    X_all = featurizer.transform(trajectories)
    assert X_all[0].shape == (n_frames, 20), (
        "unexpected shape returned: (%s, %s)" %
        X_all[0].shape)


def test_von_mises_featurizer_2():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    # test to make sure results are being put in the right order
    feat = VonMisesFeaturizer(["phi", "psi"], n_bins=10)
    _, all_phi = compute_phi(trajectories[0])
    X_all = feat.transform(trajectories)
    all_res = []
    for frame in all_phi:
        for dihedral_value in frame:
            all_res.extend(vm.pdf(dihedral_value,
                                  loc=feat.loc, kappa=feat.kappa))

    print(len(all_res))

    # this checks 10 random dihedrals to make sure that they appear in the right columns
    # for the vonmises bins
    n_phi = all_phi.shape[1]
    for k in range(5):
        # pick a random phi dihedral
        rndint = np.random.choice(range(n_phi))
        # figure out where we expect it to be in X_all
        indices_to_expect = []
        for i in range(10):
            indices_to_expect += [n_phi * i + rndint]

        # we know the results in all_res are dihedral1(bin1-bin10) dihedral2(bin1 to bin10)
        # we are checking if X is alldihedrals(bin1) then all dihedrals(bin2)

        expected_res = all_res[rndint * 10:10 + rndint * 10]

        assert (np.array(
            [X_all[0][0, i] for i in indices_to_expect]) == expected_res).all()


def test_slicer():
    X = ([np.random.normal(size=(50, 5), loc=np.arange(5))] +
         [np.random.normal(size=(10, 5), loc=np.arange(5))])

    slicer = Slicer(index=[0, 1])

    Y = slicer.transform(X)
    eq(len(Y), len(X))
    eq(Y[0].shape, (50, 2))

    slicer = Slicer(first=2)

    Y2 = slicer.transform(X)
    eq(len(Y2), len(X))
    eq(Y2[0].shape, (50, 2))

    eq(Y[0], Y2[0])
    eq(Y[1], Y2[1])
