import numpy as np

from msmbuilder.decomposition import tICA, KSparseTICA
from msmbuilder.example_datasets import MetEnkephalin
from msmbuilder.featurizer import AtomPairsFeaturizer

def build_dataset():
    trajs = MetEnkephalin().get().trajectories

    pairs = []
    for i in range(trajs[0].n_atoms):
        for j in range(i):
            pairs.append((i, j))
    np.random.seed(0)
    np.random.shuffle(pairs)
    n_pairs = 200

    return AtomPairsFeaturizer(pairs[:n_pairs]).transform([traj[::10] for traj in trajs])

def test_MetEnkephalin():
    np.random.seed(0)
    data = build_dataset()
    n_features = data[0].shape[1]

    # check whether this recovers a single 1-sparse eigenpair without error
    kstica = KSparseTICA(n_components=1, k = 1)
    _ = kstica.fit_transform(data)
    assert (np.sum(kstica.components_ != 0) == 1)

    ## check whether this recovers >1 eigenpair without error
    #kstica = KSparseTICA(n_components=2)
    #_ = kstica.fit_transform(data)

    ## check whether this recovers all eigenpairs without error
    #kstica = KSparseTICA()
    #_ = kstica.fit_transform(data)

    # check whether we recover the same solution as standard tICA when k = n_features
    n_components = 10
    kstica = KSparseTICA(n_components=n_components, k=n_features)
    tica = tICA(n_components=n_components)
    _ = kstica.fit_transform(data)
    _ = tica.fit_transform(data)
    np.testing.assert_array_almost_equal(kstica.eigenvalues_, tica.eigenvalues_)