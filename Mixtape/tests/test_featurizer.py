import numpy as np
from mdtraj.testing import eq, raises
import mixtape.featurizer
from mixtape.featurizer import subset_featurizer
from mixtape.datasets import fetch_alanine_dipeptide

def test_SubsetAtomPairs0():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = subset_featurizer.get_atompair_indices(trj0)
    featurizer = mixtape.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = subset_featurizer.SubsetAtomPairs(pair_indices, trj0)
    featurizer.subset = np.arange(len(pair_indices))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_SubsetAtomPairs1():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = subset_featurizer.get_atompair_indices(trj0)
    featurizer = mixtape.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = subset_featurizer.SubsetAtomPairs(pair_indices, trj0, subset=np.arange(len(pair_indices)))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


@raises(AssertionError)
def test_SubsetAtomPairs2():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = subset_featurizer.get_atompair_indices(trj0)
    featurizer = mixtape.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all0 = featurizer.transform(trajectories)

    featurizer = subset_featurizer.SubsetAtomPairs(pair_indices, trj0, subset=np.array([0, 1]))
    X_all = featurizer.transform(trajectories)

    any([eq(x, x0) for (x, x0) in zip(X_all, X_all0)])


def test_that_all_featurizers_run():
    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    trj0 = trajectories[0][0]
    atom_indices, pair_indices = subset_featurizer.get_atompair_indices(trj0)

    featurizer = mixtape.featurizer.AtomPairsFeaturizer(pair_indices)
    X_all = featurizer.transform(trajectories)
    
    featurizer = mixtape.featurizer.SuperposeFeaturizer(np.arange(15), trj0)
    X_all = featurizer.transform(trajectories)

    featurizer = mixtape.featurizer.DihedralFeaturizer(["phi" ,"psi"])
    X_all = featurizer.transform(trajectories)

    #featurizer = mixtape.featurizer.ContactFeaturizer()  # Doesn't work on ALA dipeptide
    #X_all = featurizer.transform(trajectories)

    featurizer = mixtape.featurizer.RMSDFeaturizer(trj0)
    X_all = featurizer.transform(trajectories)
    

    atom_featurizer0 = subset_featurizer.SubsetAtomPairs(pair_indices, trj0, exponent=-1.0)
    cosphi = subset_featurizer.SubsetCosPhiFeaturizer(trj0)
    sinphi = subset_featurizer.SubsetSinPhiFeaturizer(trj0)
    cospsi = subset_featurizer.SubsetCosPsiFeaturizer(trj0)
    sinpsi = subset_featurizer.SubsetSinPsiFeaturizer(trj0)
    
    featurizer = subset_featurizer.SubsetFeatureUnion([("pairs", atom_featurizer0), ("cosphi", cosphi), ("sinphi", sinphi), ("cospsi", cospsi), ("sinpsi", sinpsi)])
    featurizer.subsets = [np.arange(1) for i in range(featurizer.n_featurizers)]
    
    X_all = featurizer.transform(trajectories)
    eq(X_all[0].shape[1], 1 * featurizer.n_featurizers)
