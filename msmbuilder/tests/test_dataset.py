from __future__ import print_function, absolute_import, division
import os
import shutil
import tempfile

import numpy as np
from nose.tools import assert_raises
from msmbuilder.dataset import dataset
from mdtraj.testing import get_fn

from .test_commands import tempdir


def test_1():
    path = tempfile.mkdtemp()
    shutil.rmtree(path)
    try:
        X = np.random.randn(10,2)
        ds = dataset(path, 'w', 'dir-npy')
        ds[0] = X
        assert set(os.listdir(path)) == set(('PROVENANCE.txt', '00000000.npy'))
        np.testing.assert_array_equal(ds[0], X)

        assert_raises(IndexError, lambda: ds[1])
        assert len(ds) == 1

        Y = np.zeros((10, 1))
        Z = np.ones((2, 2))
        ds[1] = Y
        ds[2] = Z
        np.testing.assert_array_equal(ds[1], Y)
        np.testing.assert_array_equal(ds[2], Z)
        assert len(ds) == 3

        for i, item in enumerate(ds):
            np.testing.assert_array_equal(item, [X, Y, Z][i])
    except:
        raise
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
        ds1 = dataset(path1, 'w', 'dir-npy')
        ds1[0] = X

        ds2 = ds1.create_derived(path2)
        ds2[0] = Y

        np.testing.assert_array_equal(ds1[0], X)
        np.testing.assert_array_equal(ds2[0], Y)
        assert len(ds1) == 1
        assert len(ds2) == 1

        prov2 = ds2.provenance
        print(prov2)
        assert 2 == sum([s.startswith('  Command') for s in prov2.splitlines()])

    except:
        raise
    finally:
        shutil.rmtree(path1)
        shutil.rmtree(path2)


def test_3():
    path = tempfile.mkdtemp()
    shutil.rmtree(path)
    try:
         ds = dataset(path, 'w', 'dir-npy')
         ds[0] = np.random.randn(10,2)
         ds[1] = np.random.randn(10,2)
         ds[2] = np.random.randn(10,2)

         np.testing.assert_array_equal(ds[:][0], ds[0])
         np.testing.assert_array_equal(ds[:][1], ds[1])
         np.testing.assert_array_equal(ds[:][2], ds[2])

         np.testing.assert_array_equal(ds[1:][0], ds[1])
         np.testing.assert_array_equal(ds[1:][1], ds[2])

    finally:
        shutil.rmtree(path)


def test_4():
    path = tempfile.mkdtemp()
    shutil.rmtree(path)
    try:
         ds = dataset(path, 'w', 'dir-npy')
         ds[0] = np.random.randn(10,2)
         v = ds.get(0, mmap=True)
         assert isinstance(v, np.memmap)
         np.testing.assert_array_equal(ds[0], v)
    finally:
        shutil.rmtree(path)


def test_mdtraj_1():
    ds = dataset(get_fn('') + '*.pdb', fmt='mdtraj', verbose=True)
    print(ds.keys())
    print(ds.get(0))
    print(ds.provenance)

    ds = dataset(get_fn('') + '*.pdb', fmt='mdtraj', atom_indices=[1,2],
                 verbose=True)
    print(ds.keys())
    print(ds.get(0))
    print(ds.provenance)


def test_hdf5_1():
    with tempdir():
        ds = dataset('ds.h5', 'w', 'hdf5')
        print(ds.provenance)
        ds[0] = np.zeros(10)
        np.testing.assert_array_equal(ds.get(0), np.zeros(10))
        assert list(ds.keys()) == [0]
        assert len(ds) == 1

        ds[0] = np.random.randn(10,1)
        ds[1] = np.random.randn(10,2)
        ds[2] = np.random.randn(10,3)

        np.testing.assert_array_equal(ds[:][0], ds[0])
        np.testing.assert_array_equal(ds[:][1], ds[1])
        np.testing.assert_array_equal(ds[:][2], ds[2])

        np.testing.assert_array_equal(ds[1:][0], ds[1])
        np.testing.assert_array_equal(ds[1:][1], ds[2])

        ds.close()
        with dataset('ds.h5') as ds:
            assert ds[0].shape == (10, 1)

def test_hdf5_2():
    with tempdir():
        ds = dataset('ds.h5', 'w', 'hdf5')
        ds2 = ds.create_derived('ds2.h5')
        print(ds2.provenance)
