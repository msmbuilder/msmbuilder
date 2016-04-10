from msmbuilder.preprocessing import (Binarizer, FunctionTransformer, Imputer,
                                      KernelCenterer, LabelBinarizer,
                                      MultiLabelBinarizer, MinMaxScaler,
                                      MaxAbsScaler, Normalizer, RobustScaler,
                                      StandardScaler, PolynomialFeatures)

from sklearn.preprocessing import (Binarizer as BinarizerR,
                                   FunctionTransformer as FunctionTransformerR,
                                   Imputer as ImputerR,
                                   KernelCenterer as KernelCentererR,
                                   LabelBinarizer as LabelBinarizerR,
                                   MultiLabelBinarizer as MultiLabelBinarizerR,
                                   MinMaxScaler as MinMaxScalerR,
                                   MaxAbsScaler as MaxAbsScalerR,
                                   Normalizer as NormalizerR,
                                   RobustScaler as RobustScalerR,
                                   StandardScaler as StandardScalerR,
                                   PolynomialFeatures as PolynomialFeaturesR)

import numpy as np


random = np.random.RandomState(42)
trajs = [random.randn(10, 3) for _ in range(5)]
labels = [random.randint(low=0, high=5, size=10).reshape(-1, 1)
          for _ in range(5)]


def test_binarizer_vs_sklearn():
    # Compare msmbuilder.preprocessing.Binarizer
    # with sklearn.preprocessing.Binarizer

    binarizerr = BinarizerR()
    binarizerr.fit(np.concatenate(trajs))

    binarizer = Binarizer()
    binarizer.fit(trajs)

    y_ref1 = binarizerr.transform(trajs[0])
    y1 = binarizer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_functiontransformer_vs_sklearn():
    # Compare msmbuilder.preprocessing.FunctionTransformer
    # with sklearn.preprocessing.FunctionTransformer

    functiontransformerr = FunctionTransformerR()
    functiontransformerr.fit(np.concatenate(trajs))

    functiontransformer = FunctionTransformer()
    functiontransformer.fit(trajs)

    y_ref1 = functiontransformerr.transform(trajs[0])
    y1 = functiontransformer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_imputer_vs_sklearn():
    # Compare msmbuilder.preprocessing.Imputer
    # with sklearn.preprocessing.Imputer

    imputerr = ImputerR()
    imputerr.fit(np.concatenate(trajs))

    imputer = Imputer()
    imputer.fit(trajs)

    y_ref1 = imputerr.transform(trajs[0])
    y1 = imputer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_kernelcenterer_vs_sklearn():
    # Compare msmbuilder.preprocessing.KernelCenterer
    # with sklearn.preprocessing.KernelCenterer

    kernelcentererr = KernelCentererR()
    kernelcentererr.fit(np.concatenate(trajs))

    kernelcenterer = KernelCenterer()
    kernelcenterer.fit(trajs)

    y_ref1 = kernelcentererr.transform(trajs[0])
    y1 = kernelcenterer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_labelbinarizer_vs_sklearn():
    # Compare msmbuilder.preprocessing.LabelBinarizer
    # with sklearn.preprocessing.LabelBinarizer

    labelbinarizerr = LabelBinarizerR()
    labelbinarizerr.fit(np.concatenate(labels))

    labelbinarizer = LabelBinarizer()
    labelbinarizer.fit(labels)

    y_ref1 = labelbinarizerr.transform(labels[0])
    y1 = labelbinarizer.transform(labels)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_multilabelbinarizer_vs_sklearn():
    # Compare msmbuilder.preprocessing.MultiLabelBinarizer
    # with sklearn.preprocessing.MultiLabelBinarizer

    multilabelbinarizerr = MultiLabelBinarizerR()
    multilabelbinarizerr.fit(np.concatenate(trajs))

    multilabelbinarizer = MultiLabelBinarizer()
    multilabelbinarizer.fit(trajs)

    y_ref1 = multilabelbinarizerr.transform(trajs[0])
    y1 = multilabelbinarizer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_minmaxscaler_vs_sklearn():
    # Compare msmbuilder.preprocessing.MinMaxScaler
    # with sklearn.preprocessing.MinMaxScaler

    minmaxscalerr = MinMaxScalerR()
    minmaxscalerr.fit(np.concatenate(trajs))

    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(trajs)

    y_ref1 = minmaxscalerr.transform(trajs[0])
    y1 = minmaxscaler.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_maxabsscaler_vs_sklearn():
    # Compare msmbuilder.preprocessing.MaxAbsScaler
    # with sklearn.preprocessing.MaxAbsScaler

    maxabsscalerr = MaxAbsScalerR()
    maxabsscalerr.fit(np.concatenate(trajs))

    maxabsscaler = MaxAbsScaler()
    maxabsscaler.fit(trajs)

    y_ref1 = maxabsscalerr.transform(trajs[0])
    y1 = maxabsscaler.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_normalizer_vs_sklearn():
    # Compare msmbuilder.preprocessing.Normalizer
    # with sklearn.preprocessing.Normalizer

    normalizerr = NormalizerR()
    normalizerr.fit(np.concatenate(trajs))

    normalizer = Normalizer()
    normalizer.fit(trajs)

    y_ref1 = normalizerr.transform(trajs[0])
    y1 = normalizer.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_robustscaler_vs_sklearn():
    # Compare msmbuilder.preprocessing.RobustScaler
    # with sklearn.preprocessing.RobustScaler

    robustscalerr = RobustScalerR()
    robustscalerr.fit(np.concatenate(trajs))

    robustscaler = RobustScaler()
    robustscaler.fit(trajs)

    y_ref1 = robustscalerr.transform(trajs[0])
    y1 = robustscaler.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_standardscaler_vs_sklearn():
    # Compare msmbuilder.preprocessing.StandardScaler
    # with sklearn.preprocessing.StandardScaler

    standardscalerr = StandardScalerR()
    standardscalerr.fit(np.concatenate(trajs))

    standardscaler = StandardScaler()
    standardscaler.fit(trajs)

    y_ref1 = standardscalerr.transform(trajs[0])
    y1 = standardscaler.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)


def test_polynomialfeatures_vs_sklearn():
    # Compare msmbuilder.preprocessing.PolynomialFeatures
    # with sklearn.preprocessing.PolynomialFeatures

    polynomialfeaturesr = PolynomialFeaturesR()
    polynomialfeaturesr.fit(np.concatenate(trajs))

    polynomialfeatures = PolynomialFeatures()
    polynomialfeatures.fit(trajs)

    y_ref1 = polynomialfeaturesr.transform(trajs[0])
    y1 = polynomialfeatures.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)
