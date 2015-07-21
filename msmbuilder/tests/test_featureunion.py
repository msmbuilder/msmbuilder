from __future__ import print_function, absolute_import, division

import numpy as np
from nose.tools import assert_raises

from msmbuilder.dataset import dataset
from msmbuilder.featurizer import FeatureUnion
from .test_commands import tempdir


def test_union():
    ds1 = [None, None]
    ds2 = [None, None]

    ds1[0] = np.random.randn(10, 2)
    ds1[1] = np.random.randn(5, 2)
    ds2[0] = np.random.randn(10, 4)
    ds2[1] = np.random.randn(5, 4)
    rs1 = np.sum(ds1[0], axis=1) + np.sum(ds2[0], axis=1)
    rs2 = np.sum(ds1[1], axis=1) + np.sum(ds2[1], axis=1)

    fu = FeatureUnion(normalize=False)
    mds = fu.fit_transform((ds1, ds2))
    assert len(mds) == 2
    assert mds[0].shape == (10, 6)
    assert mds[1].shape == (5, 6)
    np.testing.assert_array_almost_equal(np.sum(mds[0], axis=1), rs1)
    np.testing.assert_array_almost_equal(np.sum(mds[1], axis=1), rs2)


def test_normalize():
    ds1 = [
        np.random.randn(10, 2),
        np.random.randn(5, 2)
    ]
    ds2 = [
        100 * np.random.randn(10, 3),
        100 * np.random.randn(5, 3)
    ]

    fu = FeatureUnion(normalize=True)
    mds = fu.fit_transform((ds1, ds2))
    assert mds[0].shape == (10, 5)
    assert mds[1].shape == (5, 5)
    combined = np.concatenate(mds[:])
    stds = np.std(combined, axis=1)
    print(stds)
    assert np.max(stds) - np.min(stds) < 2


def test_dataset():
    with tempdir():
        # This doesn't work with py2.6
        with dataset('ds1.h5', 'w', 'hdf5') as ds1, \
                dataset('ds2.h5', 'w', 'hdf5') as ds2:
            ds1[0] = np.random.randn(10, 2)
            ds1[1] = np.random.randn(5, 2)
            ds2[0] = np.random.randn(10, 4)
            ds2[1] = np.random.randn(5, 4)

            # Compare row sums
            rs1 = np.sum(ds1[0], axis=1) + np.sum(ds2[0], axis=1)
            rs2 = np.sum(ds1[1], axis=1) + np.sum(ds2[1], axis=1)

            fu = FeatureUnion(normalize=False)
            mds = fu.fit_transform((ds1, ds2))

        assert len(mds) == 2
        assert mds[0].shape == (10, 6)
        assert mds[1].shape == (5, 6)
        np.testing.assert_array_almost_equal(np.sum(mds[0], axis=1), rs1)
        np.testing.assert_array_almost_equal(np.sum(mds[1], axis=1), rs2)


def test_uneven_n():
    with tempdir():
        # This doesn't work with py2.6
        with dataset('ds1/', 'w', 'dir-npy') as ds1, \
                dataset('ds2/', 'w', 'dir-npy') as ds2:
            ds1[0] = np.random.randn(10, 2)
            ds1[1] = np.random.randn(5, 2)
            ds2[0] = np.random.randn(10, 4)
            # Uneven number of trajs!

            fu = FeatureUnion(normalize=False)
            with assert_raises(ValueError):
                fu.fit((ds1, ds2))


def test_uneven_len():
    with tempdir():
        # This doesn't work with py2.6
        with dataset('ds1/', 'w', 'dir-npy') as ds1, \
                dataset('ds2/', 'w', 'dir-npy') as ds2:
            ds1[0] = np.random.randn(10, 2)
            ds1[1] = np.random.randn(5, 2)
            ds2[0] = np.random.randn(10, 4)
            ds2[1] = np.random.randn(10, 4)
            # Uneven length!

            fu = FeatureUnion(normalize=False)
            with assert_raises(ValueError):
                fu.fit_transform((ds1, ds2))


def test_uneven_width():
    ds1 = [None, None]
    ds2 = [None, None]
    ds1[0] = np.random.randn(10, 2)
    ds1[1] = np.random.randn(5, 2)
    ds2[0] = np.random.randn(10, 4)
    ds2[1] = np.random.randn(5, 3)

    fu = FeatureUnion(normalize=True)
    with assert_raises(ValueError):
        fu.fit_transform((ds1, ds2))
