.. _preprocessing:
.. currentmodule:: msmbuilder.preprocessing


Preprocessing
=============

Preprocessing of a dataset is a common requirement for many machine learning
estimators and may involve scaling, centering, normalization, smoothing,
binarization, and imputation methods.

Preprocessors
-------------

.. autosummary::
    :toctree: _preprocessing/

    Binarizer
    Butterworth
    EWMA
    DoubleEWMA
    Imputer
    KernelCenterer
    LabelBinarizer
    MultiLabelBinarizer
    MinMaxScaler
    MaxAbsScaler
    Normalizer
    RobustScaler
    StandardScaler
    PolynomialFeatures

.. vim: tw=75
