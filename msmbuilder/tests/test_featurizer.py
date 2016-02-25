import numpy as np
from mdtraj import compute_dihedrals, compute_phi
from mdtraj.testing import eq

from msmbuilder.example_datasets import fetch_alanine_dipeptide
from msmbuilder.featurizer import get_atompair_indices, FunctionFeaturizer, \
    DihedralFeaturizer, AtomPairsFeaturizer, SuperposeFeaturizer, \
    RMSDFeaturizer, VonMisesFeaturizer, Slicer


def test_function_featurizer():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0]

    # use the dihedral to compute phi for ala
    atom_ind = [[4, 6, 8, 14]]
    func = compute_dihedrals
    # test with args
    f = FunctionFeaturizer(func, func_args={"indices": atom_ind})
    res1 = f.transform([trj0])

    # test with function in a fucntion without any args
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

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
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


def test_slicer():
    X = ([np.random.normal(size=(50, 5), loc=np.arange(5))]
         + [np.random.normal(size=(10, 5), loc=np.arange(5))])

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
