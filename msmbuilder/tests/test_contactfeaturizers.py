import numpy as np

from msmbuilder.example_datasets import MinimalFsPeptide
from msmbuilder.featurizer import BinaryContactFeaturizer
from msmbuilder.featurizer import ContactFeaturizer
from msmbuilder.featurizer import LogisticContactFeaturizer
import mdtraj as md
import itertools
from numpy.testing.decorators import skipif

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

@skipif(True)
def test_soft_min_contact_featurizer():
     #just get one frame for now
     traj = MinimalFsPeptide().get_cached().trajectories[0][0]
     soft_min_beta = 20

     ri,rj = np.random.choice(np.arange(traj.top.n_residues), size=2, replace=False)
     aind_i = [i.index for i in traj.top.residue(ri).atoms]
     aind_j = [i.index for i in traj.top.residue(rj).atoms]

     atom_pairs = [i for i in itertools.product(aind_i,aind_j)]

     featuizer = ContactFeaturizer(contacts=[[ri,rj]], scheme='closest', soft_min=True,
                                   soft_min_beta=soft_min_beta)

     features = featuizer.transform(([traj]))[0]
     distances = md.compute_distances(traj, atom_pairs)
     distances = soft_min_beta / np.log(np.sum(np.exp(soft_min_beta/distances),axis=1))

     np.allclose(features,distances)