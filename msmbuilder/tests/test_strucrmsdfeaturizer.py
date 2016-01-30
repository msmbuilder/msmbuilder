import numpy as np

from msmbuilder.example_datasets import fetch_alanine_dipeptide
from msmbuilder.featurizer import StrucRMSDFeaturizer


def test_alanine_dipeptide():
    # This test takes the rmsd of the 0th set of alanine dipeptide
    # trajectories relative to the 0th frame of the dataset.
    # The test asserts that the first rmsd calculated will be zero.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    featurizer = StrucRMSDFeaturizer(trajectories[0], trajectories[0][0],
                                     np.arange(trajectories[0].n_atoms))
    data = featurizer.transform(trajectories[0])

    assert (data[0] < 1e-3)

    # For some reason the rmsd of trajectories[0][0] with itself
    # is 0.0001041; see
    #
    #  $ ipython
    #  >>> import mdtraj as md
    #  >>> import msmbuilder.featurizer
    #  >>> from msmbuilder.example_datasets import fetch_alanine_dipeptide
    #  >>> dataset = fetch_alanine_dipeptide()
    #  >>> trajectories = dataset["trajectories"]
    #  >>> md.rmsd(trajectories[0][0],trajectories[0][0],0)


def test_two_refs():
    # This test uses the 0th and 1st frames of the 0th set of
    # adp trajectories as the two reference trajectories and
    # ensures that the rmsd of the 0th frame of the dataset with
    # the 0th reference are identical and the 1st frame of the
    # dataset with the 1st reference are identical.

    dataset = fetch_alanine_dipeptide()
    trajectories = dataset["trajectories"]
    featurizer = StrucRMSDFeaturizer(trajectories[0], trajectories[0][0:2],
                                     range(trajectories[0].n_atoms))
    data = featurizer.transform(trajectories[0])

    # TODO: Figure out why arrays are 3D
    assert (data[0][0][0] - data[1][0][1] < 1e-3)
    assert (data[1][0][0] - data[0][0][1] < 1e-3)
