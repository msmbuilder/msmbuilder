import numpy as np
from numpy.testing.decorators import skipif

from sklearn.feature_selection import (RFE as RFER,
                                       RFECV as RFECVR,
                                       VarianceThreshold as VarianceThresholdR)
try:
    from sklearn.feature_selection import SelectFromModel as SelectFromModelR
    from ..feature_selection import SelectFromModel
    HAVE_SFM = True
except ImportError:
    HAVE_SFM = False

from sklearn.svm import SVR
from sklearn.linear_model import LassoCV

from ..featurizer import DihedralFeaturizer
from ..feature_selection import FeatureSelector, RFE, RFECV, VarianceThreshold

from ..example_datasets import fetch_alanine_dipeptide as fetch_data

FEATS = [
        ('phi', DihedralFeaturizer(types=['phi'], sincos=True)),
        ('psi', DihedralFeaturizer(types=['psi'], sincos=True)),
    ]


def test_featureselector():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS, which_feat='phi')

    assert fs.which_feat == ['phi']

    y1 = fs.partial_transform(trajectories[0])
    y_ref1 = FEATS[0][1].partial_transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_rfe_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS)
    estimator = SVR()

    rfe = RFE(estimator, n_features_to_select=1)
    rfer = RFER(estimator, n_features_to_select=1)

    y = fs.partial_transform(trajectories[0])

    z1 = rfe.fit_transform([y])[0]
    z_ref1 = rfer.fit_transform(y)

    np.testing.assert_array_almost_equal(z_ref1, z1)


def test_rfecv_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS)
    estimator = SVR()

    rfecv = RFECV(estimator, step=2)
    rfecvr = RFECVR(estimator, step=2)

    y = fs.partial_transform(trajectories[0])

    z1 = rfecv.fit_transform([y])[0]
    z_ref1 = rfecvr.fit_transform(y)

    np.testing.assert_array_almost_equal(z_ref1, z1)


def test_variancethreshold_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS)

    vt = VarianceThreshold(0.1)
    vtr = VarianceThresholdR(0.1)

    y = fs.partial_transform(trajectories[0])

    z1 = vt.fit_transform([y])[0]
    z_ref1 = vtr.fit_transform(y)

    np.testing.assert_array_almost_equal(z_ref1, z1)


@skipif(not HAVE_SFM, 'this test requires sklearn >0.17.0')
def test_selectfrommodel_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    estimator = LassoCV()

    fs = FeatureSelector(FEATS)

    sfm = SelectFromModel(estimator, threshold=0.25)
    sfmr = SelectFromModelR(estimator, threshold=0.25)

    y = fs.partial_transform(trajectories[0])

    z1 = sfm.fit_transform([y])[0]
    z_ref1 = sfmr.fit_transform(y)

    np.testing.assert_array_almost_equal(z_ref1, z1)
