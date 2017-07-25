import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold as VarianceThresholdR

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.feature_selection import FeatureSelector, VarianceThreshold, FeatureSlicer
from msmbuilder.featurizer import DihedralFeaturizer

FEATS = [
    ('phi', DihedralFeaturizer(types=['phi'], sincos=True)),
    ('psi', DihedralFeaturizer(types=['psi'], sincos=True)),
]


def test_featureselector_order():
    fs1 = FeatureSelector(FEATS)
    fs2 = FeatureSelector(FEATS[::-1])

    assert fs1.which_feat == ['phi', 'psi']
    assert fs2.which_feat == ['psi', 'phi']


def test_featureselector_selection():
    trajectories = AlanineDipeptide().get_cached().trajectories
    fs = FeatureSelector(FEATS, which_feat='phi')

    assert fs.which_feat == ['phi']

    y1 = fs.partial_transform(trajectories[0])
    y_ref1 = FEATS[0][1].partial_transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_featureselector_transform():
    trajectories = AlanineDipeptide().get_cached().trajectories
    fs = FeatureSelector(FEATS, which_feat='psi')
    y1 = fs.transform(trajectories)
    assert len(y1) == len(trajectories)


def test_variancethreshold_vs_sklearn():
    trajectories = AlanineDipeptide().get_cached().trajectories
    fs = FeatureSelector(FEATS)

    vt = VarianceThreshold(0.1)
    vtr = VarianceThresholdR(0.1)

    y = fs.partial_transform(trajectories[0])

    z1 = vt.fit_transform([y])[0]
    z_ref1 = vtr.fit_transform(y)

    np.testing.assert_array_almost_equal(z_ref1, z1)

def test_which_feat_types():
    # trajectories = AlanineDipeptide().get_cached().trajectories
    fs = FeatureSelector(FEATS, which_feat=('phi', 'psi'))
    assert fs.which_feat == ['phi', 'psi']

    fs = FeatureSelector(FEATS, which_feat=set(('phi', 'psi')))
    assert fs.which_feat == ['phi', 'psi'] or fs.which_feat == ['psi', 'phi']

    try:
        fs = FeatureSelector(FEATS, which_feat={'phi':'psi'})
        assert False
    except TypeError:
        pass

    try:
        fs = FeatureSelector(FEATS, which_feat=['phiii'])
        assert False
    except ValueError:
        pass



def test_feature_slicer():
    trajectories = AlanineDipeptide().get_cached().trajectories
    f = DihedralFeaturizer()
    fs = FeatureSlicer(f, indices=[0,1])
    y1 = fs.transform(trajectories)
    assert y1[0].shape[1] == 2

    df = pd.DataFrame(fs.describe_features(trajectories[0]))
    assert len(df) == 2
    assert 'psi' not in df.featuregroup[0]
    assert 'psi' not in df.featuregroup[1]

    fs = FeatureSlicer(f, indices=[2,3])
    y1 = fs.transform(trajectories)
    assert y1[0].shape[1] == 2

    df = pd.DataFrame(fs.describe_features(trajectories[0]))
    assert len(df) == 2
    assert 'phi' not in df.featuregroup[0]
    assert 'phi' not in df.featuregroup[1]
