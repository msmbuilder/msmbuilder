import os
import shutil
import tempfile

import numpy as np
from nose.tools import assert_raises
from mixtape.dataset import dataset


def test_1():
    path = tempfile.mkdtemp()
    shutil.rmtree(path)
    try:
        X = np.random.randn(10,2)
        ds = dataset(path, 'w')
        ds.append(X)
        assert set(os.listdir(path)) == set(('PROVENANCE.txt', '00000000.npy'))
        np.testing.assert_array_equal(ds[0], X)

        assert_raises(IndexError, lambda: ds[1])
        assert len(ds) == 1

        Y = np.zeros((10, 1))
        Z = np.ones((2, 2))
        ds.extend([Y, Z])
        np.testing.assert_array_equal(ds[1], Y)
        np.testing.assert_array_equal(ds[2], Z)
        assert len(ds) == 3

        for i, item in enumerate(ds):
            np.testing.assert_array_equal(item, [X, Y, Z][i])

    finally:
        shutil.rmtree(path)


def test_2():
    path1 = tempfile.mkdtemp()
    path2 = tempfile.mkdtemp()
    shutil.rmtree(path1)
    shutil.rmtree(path2)
    try:

        X = np.random.randn(10,2)
        Y = np.random.randn(10,2)
        ds1 = dataset(path1, 'w')
        ds1.append(X)

        ds2 = ds1.save_transformed(path2, [Y])

        np.testing.assert_array_equal(ds1[0], X)
        np.testing.assert_array_equal(ds2[0], Y)
        assert len(ds1) == 1
        assert len(ds2) == 1

        prov2 = ds2.provenance()
        assert 2 == sum([s.startswith('Command') for s in prov2.splitlines()])

    finally:
        shutil.rmtree(path1)
        shutil.rmtree(path2)
