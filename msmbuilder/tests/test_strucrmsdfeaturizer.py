import numpy as np
import mdtraj as md
from msmbuilder.example_datasets import fetch_alanine_dipeptide
from msmbuilder.featurizer import StrucRMSDFeaturizer

# np.testing.assert_array_almost_equal(array1,array2)

def test_alanine_dipeptide_basic():
    # This test takes the rmsd of the 0th set of alanine dipeptide
    # trajectories relative to the 0th frame of the dataset.
    # The test asserts that all rmsd's calculated will be equal
    # to the ones that would be calculated straight from mdtraj.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    featurizer = StrucRMSDFeaturizer(trajectories[0][0])
    data = featurizer.transform(trajectories[0:1])

    true_rmsd = md.rmsd(trajectories[0], trajectories[0][0])

    np.testing.assert_array_almost_equal(data[0][:,0], true_rmsd, decimal=4)

def test_omitting_indices():
    # This test verifies that the result produced when
    # atom_indices are omitted is the same as the result
    # produced when atom_indices is all atom indices.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]

    featurizer_indices = StrucRMSDFeaturizer(trajectories[0][0],
                                    np.arange(trajectories[0].n_atoms))
    data_indices = featurizer_indices.transform(trajectories[0:1])
    featurizer = StrucRMSDFeaturizer(trajectories[0][0])
    data = featurizer.transform(trajectories[0:1])

    np.testing.assert_array_almost_equal(data[0][:,0],
                    data_indices[0][:,0], decimal=4)

def test_different_indices():
    # This test verifies that the rmsd's calculated from
    # different sets of atom indices are not the same,
    # but that the arrays are still the same shape.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    n_atoms = trajectories[0].n_atoms
    halfway_point = n_atoms//2

    featurizer_first_half = StrucRMSDFeaturizer(trajectories[0][0],
                                    np.arange(halfway_point))
    data_first_half = featurizer_first_half.transform(trajectories[0:1])
    featurizer_second_half = StrucRMSDFeaturizer(trajectories[0][0],
                                    np.arange(halfway_point,n_atoms))
    data_second_half = featurizer_second_half.transform(trajectories[0:1])

    assert data_first_half[0].shape == data_second_half[0].shape
    # janky way to show that the arrays shouldn't be equal here
    assert sum(data_first_half[0][:,0]) != sum(data_second_half[0][:,0])


def test_two_refs_basic():
    # This test uses the 0th and 1st frames of the 0th set of
    # adp trajectories as the two reference trajectories and
    # ensures that the rmsd of the 0th frame of the dataset with
    # the 0th reference are identical and the 1st frame of the
    # dataset with the 1st reference are identical.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    featurizer = StrucRMSDFeaturizer(trajectories[0][0:2])
    data = featurizer.transform(trajectories[0:1])

    true_rmsd = np.zeros((trajectories[0].n_frames, 2))
    for frame in range(2):
        true_rmsd[:, frame] = md.rmsd(trajectories[0], trajectories[0][frame])

    np.testing.assert_almost_equal(data[0][0,0], data[0][1,1], decimal=3)
    np.testing.assert_almost_equal(data[0][1,0], data[0][0,1], decimal=3)

    np.testing.assert_array_almost_equal(data[0], true_rmsd, decimal=4)


def test_two_refs_omitting_indices():
    # This test verifies that the result produced when
    # atom_indices are omitted is the same as the result
    # produced when atom_indices is all atom indices.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    featurizer_indices = StrucRMSDFeaturizer(trajectories[0][0:2],
                                    np.arange(trajectories[0].n_atoms))
    data_indices = featurizer_indices.transform(trajectories[0:1])

    featurizer = StrucRMSDFeaturizer(trajectories[0][0:2])
    data = featurizer.transform(trajectories[0:1])

    np.testing.assert_array_almost_equal(data[0], data_indices[0], decimal=4)


def test_two_refs_different_indices():
    # This test verifies that the rmsd's calculated from
    # different sets of atom indices are not the same,
    # but that the arrays are still the same shape.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    n_atoms = trajectories[0].n_atoms
    halfway_point = n_atoms//2

    featurizer_first_half = StrucRMSDFeaturizer(trajectories[0][0:2],
                                    np.arange(halfway_point))
    data_first_half = featurizer_first_half.transform(trajectories[0:1])
    featurizer_second_half = StrucRMSDFeaturizer(trajectories[0][0:2],
                                    np.arange(halfway_point,n_atoms))
    data_second_half = featurizer_second_half.transform(trajectories[0:1])

    assert data_first_half[0].shape == data_second_half[0].shape
    # janky way to show that the arrays shouldn't be equal here
    assert sum(data_first_half[0][:,0]) != sum(data_second_half[0][:,0])
    assert sum(data_first_half[0][:,1]) != sum(data_second_half[0][:,1])


