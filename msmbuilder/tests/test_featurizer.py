import numpy as np
from mdtraj.testing import eq

import msmbuilder.featurizer
from msmbuilder.example_datasets import fetch_alanine_dipeptide
from msmbuilder.featurizer import get_atompair_indices, SubsetAtomPairs, \
    SubsetCosPhiFeaturizer, SubsetCosPsiFeaturizer, SubsetFeatureUnion, \
    SubsetSinPhiFeaturizer, SubsetSinPsiFeaturizer


def test_SubsetAtomPairs_1():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = msmbuilder.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = SubsetAtomPairs(pair_indices, trj0)
    featurizer.subset = np.arange(len(pair_indices))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_SubsetAtomPairs_2():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = msmbuilder.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = SubsetAtomPairs(pair_indices, trj0,
                                 subset=np.arange(len(pair_indices)))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_SubsetAtomPairs_3():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = msmbuilder.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = SubsetAtomPairs(pair_indices, trj0, subset=np.array([0, 1]))
    X_all = featurizer.transform(trajectories)

    try:
        any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])
    except AssertionError:
        pass
    else:
        raise AssertionError("Did not raise an assertion!")


def test_that_all_featurizers_run():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)

    featurizer = msmbuilder.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all = featurizer.transform(trajectories)

    featurizer = msmbuilder.featurizer.SuperposeFeaturizer(np.arange(15), trj0)
    X_all = featurizer.transform(trajectories)

    featurizer = msmbuilder.featurizer.DihedralFeaturizer(["phi", "psi"])
    X_all = featurizer.transform(trajectories)

    # Below doesn't work on ALA dipeptide
    # featurizer = msmbuilder.featurizer.ContactFeaturizer()
    # X_all = featurizer.transform(trajectories)

    featurizer = msmbuilder.featurizer.RMSDFeaturizer(trj0)
    X_all = featurizer.transform(trajectories)

    atom_featurizer0 = SubsetAtomPairs(pair_indices, trj0, exponent=-1.0)
    cosphi = SubsetCosPhiFeaturizer(trj0)
    sinphi = SubsetSinPhiFeaturizer(trj0)
    cospsi = SubsetCosPsiFeaturizer(trj0)
    sinpsi = SubsetSinPsiFeaturizer(trj0)

    featurizer = SubsetFeatureUnion([
        ("pairs", atom_featurizer0), ("cosphi", cosphi),
        ("sinphi", sinphi), ("cospsi", cospsi), ("sinpsi", sinpsi)
    ])
    featurizer.subsets = [np.arange(1) for i in range(featurizer.n_featurizers)]

    X_all = featurizer.transform(trajectories)
    eq(X_all[0].shape[1], 1 * featurizer.n_featurizers)


def test_slicer():
    X = ([np.random.normal(size=(50, 5), loc=np.arange(5))]
         + [np.random.normal(size=(10, 5), loc=np.arange(5))])

    slicer = msmbuilder.featurizer.Slicer(index=[0, 1])

    Y = slicer.transform(X)
    eq(len(Y), len(X))
    eq(Y[0].shape, (50, 2))

    slicer = msmbuilder.featurizer.FirstSlicer(first=2)

    Y2 = slicer.transform(X)
    eq(len(Y2), len(X))
    eq(Y2[0].shape, (50, 2))

    eq(Y[0], Y2[0])
    eq(Y[1], Y2[1])
