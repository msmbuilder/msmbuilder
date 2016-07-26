import numpy as np

from msmbuilder.example_datasets import MinimalFsPeptide
from msmbuilder.featurizer import BinaryContactFeaturizer
from msmbuilder.featurizer import ContactFeaturizer
from msmbuilder.featurizer import LogisticContactFeaturizer


def test_contacts():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    contactfeaturizer = ContactFeaturizer()
    contacts = contactfeaturizer.transform(trajectories)

    assert contacts[0].shape[1] == 171


def test_binaries():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    binarycontactfeaturizer = BinaryContactFeaturizer()
    binaries = binarycontactfeaturizer.transform(trajectories)

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) <= binaries[0].shape[0] * binaries[0].shape[1]


def test_binaries_inf_cutoff():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=1e10)
    binaries = binarycontactfeaturizer.transform(trajectories)

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) == binaries[0].shape[0] * binaries[0].shape[1]


def test_binaries_zero_cutoff():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=0)
    binaries = binarycontactfeaturizer.transform(trajectories)

    assert binaries[0].shape[1] == 171
    assert np.sum(binaries[0]) == 0


def test_logistics():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    logisticcontactfeaturizer = LogisticContactFeaturizer()
    logistics = logisticcontactfeaturizer.transform(trajectories)

    assert logistics[0].shape[1] == 171
    assert np.amax(logistics[0]) < 1.0
    assert np.amin(logistics[0]) > 0.0


def test_distance_to_logistic():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    steepness = np.absolute(10 * np.random.randn())
    center = np.absolute(np.random.randn())
    contactfeaturizer = ContactFeaturizer()
    contacts = contactfeaturizer.transform(trajectories)
    logisticcontactfeaturizer = LogisticContactFeaturizer(center=center,
                                                          steepness=steepness)
    logistics = logisticcontactfeaturizer.transform(trajectories)

    for n in range(10):
        i = np.random.randint(0, contacts[0].shape[0] - 1)
        j = np.random.randint(0, contacts[0].shape[1] - 1)

        x = contacts[0][i][j]
        y = logistics[0][i][j]

        if (x > center):
            assert y < 0.5
        if (x < center):
            assert y > 0.5


def test_binary_to_logistics():
    trajectories = MinimalFsPeptide().get_cached().trajectories
    steepness = np.absolute(10 * np.random.randn())
    center = np.absolute(np.random.randn())
    binarycontactfeaturizer = BinaryContactFeaturizer(cutoff=center)
    binaries = binarycontactfeaturizer.transform(trajectories)
    logisticcontactfeaturizer = LogisticContactFeaturizer(center=center,
                                                          steepness=steepness)
    logistics = logisticcontactfeaturizer.transform(trajectories)

    # This checks that no distances that are larger than the center are logistically
    # transformed such that they are less than 1/2
    np.testing.assert_array_almost_equal(binaries[0], logistics[0] > 0.5)
