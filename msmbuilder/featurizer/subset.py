import mdtraj as md
import numpy as np

from . import Featurizer, TrajFeatureUnion


class BaseSubsetFeaturizer(Featurizer):
    """Base class for featurizers that have a subset of active features.
    n_features refers to the number of active features.  n_max refers to the
    number of possible features.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        Reference Trajectory for checking consistency
    subset : np.ndarray, default=None, dtype=int
        The values in subset specify which of all possible features

    Notes
    -----

    As an example, suppose we have an instance that has `n_max` = 5.  This
    means that the possible features are subsets of [0, 1, 2, 3, 4].  One possible
    subset is then [0, 1, 3].  The allowed values of subset (e.g. `n_max`)
    will be determined by the subclass--e.g. for example, `n_max` might be
    the number of phi backbone angles.
    """

    def __init__(self, reference_traj, subset=None):
        self.reference_traj = reference_traj
        if subset is not None:
            self.subset = subset
        else:
            self.subset = np.zeros(0, 'int')

    @property
    def n_features(self):
        return len(self.subset)


class SubsetAtomPairs(BaseSubsetFeaturizer):
    """Subset featurizer based on atom pair distances.

    Parameters
    ----------
    possible_pair_indices : np.ndarray, dtype=int, shape=(n_max, 2)
        These are the possible atom indices to use for calculating interatomic
        distances.
    reference_traj : mdtraj.Trajectory
        Reference Trajectory for checking consistency
    subset : np.ndarray, default=None, dtype=int
        The values in subset specify which of all possible features are
        to be enabled.  Specifically, atom pair distances are calculated
        for the pairs `possible_pair_indices[subset]`
    periodic : bool, optional, default=False
        if True, use periodic boundary condition wrapping
    exponent : float, optional, default=1.0
        Use the distances to this power as the output feature.

    See Also
    --------

    See `get_atompair_indices` for how one might generate acceptable atom pair
    indices.

    """
    def __init__(self, possible_pair_indices, reference_traj, subset=None, periodic=False, exponent=1.0):
        super(SubsetAtomPairs, self).__init__(reference_traj, subset=subset)
        self.possible_pair_indices = possible_pair_indices
        self.periodic = periodic
        self.exponent = exponent
        if subset is None:
            self.subset = np.zeros(0, 'int')
        else:
            self.subset = subset


    @property
    def n_max(self):
        return len(self.possible_pair_indices)

    def partial_transform(self, traj):
        if self.n_features > 0:
            features = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic) ** self.exponent
        else:
            features = np.zeros((traj.n_frames, 0))
        return features

    @property
    def pair_indices(self):
        return self.possible_pair_indices[self.subset]


class SubsetTrigFeaturizer(BaseSubsetFeaturizer):
    """Base class for featurizer based on dihedral sine or cosine.

    Notes
    -----

    Subsets must be a subset of 0, ..., n_max - 1, where n_max is determined
    by the number of respective phi / psi dihedrals in your protein, as
    calcualted by mdtraj.compute_phi and mdtraj.compute_psi

    """

    def partial_transform(self, traj):
        if self.n_features > 0:
            dih = md.geometry.dihedral.compute_dihedrals(traj, self.which_atom_ind[self.subset])
            features = self.trig_function(dih)
        else:
            features = np.zeros((traj.n_frames, 0))
        return features

    @property
    def n_max(self):
        return len(self.which_atom_ind)

class CosMixin(object):
    def trig_function(self, dihedrals):
        return np.cos(dihedrals)

class SinMixin(object):
    def trig_function(self, dihedrals):
        return np.sin(dihedrals)

class PhiMixin(object):
    @property
    def which_atom_ind(self):
        atom_indices, dih = md.geometry.dihedral.compute_phi(self.reference_traj)
        return atom_indices

class PsiMixin(object):
    @property
    def which_atom_ind(self):
        atom_indices, dih = md.geometry.dihedral.compute_psi(self.reference_traj)
        return atom_indices


class SubsetCosPhiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass


class SubsetCosPsiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass


class SubsetSinPhiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass


class SubsetSinPsiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass


class SubsetFeatureUnion(TrajFeatureUnion):
    """MSMBuilder version of sklearn.pipeline.FeatureUnion with feature subset selection.

    Notes
    -----
    Works on lists of trajectories.
    Has a hacky convenience method to set all subsets at once.
    """

    @property
    def subsets(self):
        return [featurizer.subset for (_, featurizer) in self.transformer_list]

    @subsets.setter
    def subsets(self, value):
        assert len(value) == len(self.transformer_list), "wrong len"
        for k, (_, featurizer) in enumerate(self.transformer_list):
            featurizer.subset = value[k]


    @property
    def n_max_i(self):
        return np.array([featurizer.n_max for (_, featurizer) in self.transformer_list])

    @property
    def n_features_i(self):
        return np.array([featurizer.n_features for (_, featurizer) in self.transformer_list])

    @property
    def n_featurizers(self):
        return len(self.transformer_list)

    @property
    def n_max(self):
        return np.sum([featurizer.n_max for (_, featurizer) in self.transformer_list])

    @property
    def n_features(self):
        return sum([featurizer.n_features for (_, featurizer) in self.transformer_list])





class DummyCV(object):
    """A cross-validation object that returns identical training and test sets."""
    def __init__(self, n):
        self.n = n

    def __iter__(self):
            yield np.arange(self.n), np.arange(self.n)

    def __len__(self):
        return self.n
