import numpy as np
from mdtraj.testing import eq

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import AtomPairsFeaturizer, get_atompair_indices
from msmbuilder.featurizer.subset import SubsetAtomPairs, \
    SubsetCosPhiFeaturizer, SubsetCosPsiFeaturizer, SubsetFeatureUnion, \
    SubsetSinPhiFeaturizer, SubsetSinPsiFeaturizer


def test_SubsetAtomPairs_1():
    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = SubsetAtomPairs(pair_indices, trj0)
    featurizer.subset = np.arange(len(pair_indices))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_SubsetAtomPairs_2():
    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = SubsetAtomPairs(pair_indices, trj0,
                                 subset=np.arange(len(pair_indices)))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_SubsetAtomPairs_3():
    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)
    featurizer = AtomPairsFeaturizer(pair_indices)
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
    trajectories = AlanineDipeptide().get_cached().trajectories
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = get_atompair_indices(trj0)

    atom_featurizer0 = SubsetAtomPairs(pair_indices, trj0, exponent=-1.0)
    cosphi = SubsetCosPhiFeaturizer(trj0)
    sinphi = SubsetSinPhiFeaturizer(trj0)
    cospsi = SubsetCosPsiFeaturizer(trj0)
    sinpsi = SubsetSinPsiFeaturizer(trj0)

    featurizer = SubsetFeatureUnion([
        ("pairs", atom_featurizer0),
        ("cosphi", cosphi),
        ("sinphi", sinphi),
        ("cospsi", cospsi),
        ("sinpsi", sinpsi)
    ])
    featurizer.subsets = [np.arange(1) for i in range(featurizer.n_featurizers)]

    X_all = featurizer.transform(trajectories)
    eq(X_all[0].shape[1], 1 * featurizer.n_featurizers)
