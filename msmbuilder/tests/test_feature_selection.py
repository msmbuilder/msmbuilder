import numpy as np

from ..featurizer import DihedralFeaturizer
from ..feature_selection import FeatureSelector

from ..example_datasets import fetch_fs_peptide

FEATS = [
        ('phi', DihedralFeaturizer(types=['phi'], sincos=True)),
        ('psi', DihedralFeaturizer(types=['psi'], sincos=True)),
    ]


def test_featureselector():
    dataset = fetch_fs_peptide()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS)

    assert fs.which_feat == 'phi'

    y1 = fs.partial_transform(trajectories[0])
    y_ref1 = FEATS[0][1].partial_transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)
