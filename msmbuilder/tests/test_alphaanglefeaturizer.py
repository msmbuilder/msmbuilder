import warnings

import msmbuilder.featurizer
from msmbuilder.example_datasets import MinimalFsPeptide, AlanineDipeptide

warnings.filterwarnings('ignore', message='.*Unlikely unit cell vectors.*')


def test_alanine_dipeptide():
    # will produce 0 features because not enough peptides

    trajectories = AlanineDipeptide().get_cached().trajectories
    featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer()
    nothing = featurizer.transform(trajectories)

    assert (nothing[0].shape[1] == 0)


def test_fs_peptide():
    # will produce 36 features

    trajectories = MinimalFsPeptide().get_cached().trajectories
    featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer()
    alphas = featurizer.transform(trajectories)

    assert (alphas[0].shape[1] == 36)


def test_fs_peptide_nosincos():
    # will produce 18 features

    trajectories = MinimalFsPeptide().get_cached().trajectories
    featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer(sincos=False)
    alphas = featurizer.transform(trajectories)

    assert (alphas[0].shape[1] == 18)
