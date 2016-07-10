import warnings

import mdtraj as md
import numpy as np

from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import Featurizer
from msmbuilder.featurizer import RMSDFeaturizer


class OldRMSDFeaturizer(Featurizer):
    """Featurizer based on RMSD to a series of reference frames.

    Parameters
    ----------
    trj0 : mdtraj.Trajectory
        Reference trajectory.  trj0.n_frames gives the number of features
        in this Featurizer.
    atom_indices : np.ndarray, default=None
        Which atom indices to use during RMSD calculation.  If None, MDTraj
        should default to all atoms.

    Notes
    -----
    This was the RMSDFeaturizer until version 3.4

    """

    def __init__(self, trj0, atom_indices=None):
        self.n_features = trj0.n_frames
        self.trj0 = trj0
        self.atom_indices = atom_indices

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space by calculating
        the RMSD to each frame in a reference trajectory.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        X = np.zeros((traj.n_frames, self.n_features))

        for frame in range(self.n_features):
            X[:, frame] = md.rmsd(traj, self.trj0,
                                  atom_indices=self.atom_indices,
                                  frame=frame)
        return X


def test_alanine_dipeptide_basic():
    # This test takes the rmsd of the 0th set of alanine dipeptide
    # trajectories relative to the 0th frame of the dataset.
    # The test asserts that all rmsd's calculated will be equal
    # to the ones that would be calculated straight from mdtraj.

    trajectories = AlanineDipeptide().get_cached().trajectories
    featurizer = RMSDFeaturizer(trajectories[0][0])
    data = featurizer.transform(trajectories[0:1])

    true_rmsd = md.rmsd(trajectories[0], trajectories[0][0])

    np.testing.assert_array_almost_equal(data[0][:, 0], true_rmsd, decimal=4)


def test_omitting_indices():
    # This test verifies that the result produced when
    # atom_indices are omitted is the same as the result
    # produced when atom_indices is all atom indices.

    trajectories = AlanineDipeptide().get_cached().trajectories

    featurizer_indices = RMSDFeaturizer(trajectories[0][0],
                                        np.arange(trajectories[0].n_atoms))
    data_indices = featurizer_indices.transform(trajectories[0:1])
    featurizer = RMSDFeaturizer(trajectories[0][0])
    data = featurizer.transform(trajectories[0:1])

    np.testing.assert_array_almost_equal(data[0][:, 0],
                                         data_indices[0][:, 0], decimal=4)


def test_different_indices():
    # This test verifies that the rmsd's calculated from
    # different sets of atom indices are not the same,
    # but that the arrays are still the same shape.

    trajectories = AlanineDipeptide().get_cached().trajectories
    n_atoms = trajectories[0].n_atoms
    halfway_point = n_atoms // 2

    featurizer_first_half = RMSDFeaturizer(trajectories[0][0],
                                           np.arange(halfway_point))
    data_first_half = featurizer_first_half.transform(trajectories[0:1])
    featurizer_second_half = RMSDFeaturizer(trajectories[0][0],
                                            np.arange(halfway_point, n_atoms))
    data_second_half = featurizer_second_half.transform(trajectories[0:1])

    assert data_first_half[0].shape == data_second_half[0].shape
    # janky way to show that the arrays shouldn't be equal here
    assert sum(data_first_half[0][:, 0]) != sum(data_second_half[0][:, 0])


def test_two_refs_basic():
    # This test uses the 0th and 1st frames of the 0th set of
    # adp trajectories as the two reference trajectories and
    # ensures that the rmsd of the 0th frame of the dataset with
    # the 0th reference are identical and the 1st frame of the
    # dataset with the 1st reference are identical.

    trajectories = AlanineDipeptide().get_cached().trajectories
    featurizer = RMSDFeaturizer(trajectories[0][0:2])
    data = featurizer.transform(trajectories[0:1])

    true_rmsd = np.zeros((trajectories[0].n_frames, 2))
    for frame in range(2):
        true_rmsd[:, frame] = md.rmsd(trajectories[0], trajectories[0][frame])

    np.testing.assert_almost_equal(data[0][0, 0], data[0][1, 1], decimal=3)
    np.testing.assert_almost_equal(data[0][1, 0], data[0][0, 1], decimal=3)

    np.testing.assert_array_almost_equal(data[0], true_rmsd, decimal=4)


def test_two_refs_omitting_indices():
    # This test verifies that the result produced when
    # atom_indices are omitted is the same as the result
    # produced when atom_indices is all atom indices.

    trajectories = AlanineDipeptide().get_cached().trajectories
    featurizer_indices = RMSDFeaturizer(trajectories[0][0:2],
                                        np.arange(trajectories[0].n_atoms))
    data_indices = featurizer_indices.transform(trajectories[0:1])

    featurizer = RMSDFeaturizer(trajectories[0][0:2])
    data = featurizer.transform(trajectories[0:1])

    np.testing.assert_array_almost_equal(data[0], data_indices[0], decimal=4)


def test_two_refs_different_indices():
    # This test verifies that the rmsd's calculated from
    # different sets of atom indices are not the same,
    # but that the arrays are still the same shape.

    trajectories = AlanineDipeptide().get_cached().trajectories
    n_atoms = trajectories[0].n_atoms
    halfway_point = n_atoms // 2

    featurizer_first_half = RMSDFeaturizer(trajectories[0][0:2],
                                           np.arange(halfway_point))
    data_first_half = featurizer_first_half.transform(trajectories[0:1])
    featurizer_second_half = RMSDFeaturizer(trajectories[0][0:2],
                                            np.arange(halfway_point, n_atoms))
    data_second_half = featurizer_second_half.transform(trajectories[0:1])

    assert data_first_half[0].shape == data_second_half[0].shape
    # janky way to show that the arrays shouldn't be equal here
    assert sum(data_first_half[0][:, 0]) != sum(data_second_half[0][:, 0])
    assert sum(data_first_half[0][:, 1]) != sum(data_second_half[0][:, 1])


def _random_trajs():
    top = md.Topology()
    c = top.add_chain()
    r = top.add_residue('HET', c)
    for _ in range(101):
        top.add_atom('CA', md.element.carbon, r)
    traj1 = md.Trajectory(xyz=np.random.uniform(size=(100, 101, 3)),
                          topology=top,
                          time=np.arange(100))
    traj2 = md.Trajectory(xyz=np.random.uniform(size=(100, 101, 3)),
                          topology=top,
                          time=np.arange(100))
    ref = md.Trajectory(xyz=np.random.uniform(size=(7, 101, 3)),
                        topology=top,
                        time=np.arange(7))
    return traj1, traj2, ref


def test_api_still_works_names():
    traj1, traj2, ref = _random_trajs()
    old = OldRMSDFeaturizer(trj0=ref, atom_indices=np.arange(50))
    with warnings.catch_warnings(record=True) as w:
        new = RMSDFeaturizer(trj0=ref, atom_indices=np.arange(50))
        assert "deprecated" in str(w[-1].message)
        assert "trj0" in str(w[-1].message)

    data_old = old.fit_transform([traj1, traj2])
    data_new = new.fit_transform([traj1, traj2])

    for do, dn in zip(data_old, data_new):
        np.testing.assert_array_almost_equal(do, dn)
        assert dn.shape == (100, 7)


def test_api_still_works_order():
    traj1, traj2, ref = _random_trajs()
    old = OldRMSDFeaturizer(ref, atom_indices=np.arange(50))
    new = RMSDFeaturizer(ref, atom_indices=np.arange(50))

    data_old = old.fit_transform([traj1, traj2])
    data_new = new.fit_transform([traj1, traj2])

    for do, dn in zip(data_old, data_new):
        np.testing.assert_array_almost_equal(do, dn)
        assert dn.shape == (100, 7)


def test_api_still_works_allframes():
    traj1, traj2, ref = _random_trajs()
    old = OldRMSDFeaturizer(ref)
    new = RMSDFeaturizer(ref)

    data_old = old.fit_transform([traj1, traj2])
    data_new = new.fit_transform([traj1, traj2])

    for do, dn in zip(data_old, data_new):
        np.testing.assert_array_almost_equal(do, dn)
        assert dn.shape == (100, 7)
