import numpy as np
from numpy.testing.decorators import skipif

from sklearn.pipeline import FeatureUnion as FeatureUnionR
from sklearn.feature_selection import (RFE as RFER, RFECV as RFECVR,
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
from ..feature_selection import (FeatureSelector, FeatureUnion,
                                 RFE, RFECV, VarianceThreshold)

from ..example_datasets import fetch_alanine_dipeptide as fetch_data

FEATS = [
        ('phi', DihedralFeaturizer(types=['phi'], sincos=True)),
        ('psi', DihedralFeaturizer(types=['psi'], sincos=True)),
    ]


def test_featureselector():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    fs = FeatureSelector(FEATS)

    assert fs.which_feat == 'phi'

    y1 = fs.partial_transform(trajectories[0])
    y_ref1 = FEATS[0][1].partial_transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_featureunion_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    transformer_weights = {'phi': 0.4, 'psi': 0.6}

    fu = FeatureUnion(FEATS, transformer_weights=transformer_weights)
    fur = FeatureUnionR(FEATS, transformer_weights=transformer_weights)

    y1 = fu.transform([trajectories[0]])[0]
    y_ref1 = fur.transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_rfe_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    estimator = SVR()

    rfe = RFE(estimator, n_features_to_select=1)
    rfer = RFER(estimator, n_features_to_select=1)

    y1 = rfe.transform([trajectories[0]])[0]
    y_ref1 = rfer.transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_rfecv_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    estimator = SVR()

    rfe = RFECV(estimator, step=2)
    rfer = RFECVR(estimator, step=2)

    y1 = rfe.transform([trajectories[0]])[0]
    y_ref1 = rfer.transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_variancethreshold_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    rfe = VarianceThreshold(0.1)
    rfer = VarianceThresholdR(0.1)

    y1 = rfe.transform([trajectories[0]])[0]
    y_ref1 = rfer.transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)


@skipif(not HAVE_SFM, 'this test requires sklearn >0.17.0')
def test_selectfrommodel_vs_sklearn():
    dataset = fetch_data()
    trajectories = dataset["trajectories"]

    estimator = LassoCV()

    rfe = SelectFromModel(estimator, threshold=0.25)
    rfer = SelectFromModelR(estimator, threshold=0.25)

    y1 = rfe.transform([trajectories[0]])[0]
    y_ref1 = rfer.transform(trajectories[0])

    np.testing.assert_array_almost_equal(y_ref1, y1)
