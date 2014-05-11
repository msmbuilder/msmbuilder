from contextlib import contextmanager
import itertools
import numpy as np
import mixtape.featurizer, mixtape.tica
import mdtraj as md
import sklearn.pipeline

ATOM_NAMES = ["N", "CA", "CB", "C", "O", "H"]

def get_atompair_indices(reference_traj, keep_atoms=ATOM_NAMES, exclude_atoms=None, reject_bonded=True):
    """Get a list of acceptable atom pairs.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        Trajectory to grab atom pairs from
    keep_atoms : np.ndarray, dtype=string, optional
        Select only these atom names
    exclude_atoms : np.ndarray, dtype=string, optional
        Exclude these atom names
    reject_bonded : bool, default=True
        If True, exclude bonded atompairs.  
        
    Returns
    -------
    atom_indices : np.ndarray, dtype=int
        The atom indices that pass your criteria
    pair_indices : np.ndarray, dtype=int, shape=(N, 2)
        Pairs of atom indices that pass your criteria.

    Notes
    -----
    This function has been optimized for speed.  A naive implementation
    can be slow (~minutes) for large proteins.
    """
    top, bonds = reference_traj.top.to_dataframe()
    
    if keep_atoms is not None:
        atom_indices = top[top.name.isin(keep_atoms) == True].index.values
    
    if exclude_atoms is not None:
        atom_indices = top[top.name.isin(exclude_atoms) == False].index.values

    pair_indices = np.array(list(itertools.combinations(atom_indices, 2)))

    if reject_bonded:
        a_list = bonds.min(1)
        b_list = bonds.max(1)

        n = atom_indices.max() + 1

        bond_hashes = a_list + b_list * n
        pair_hashes = pair_indices[:, 0] + pair_indices[:,1] * n

        not_bonds = ~np.in1d(pair_hashes, bond_hashes)

        pair_indices = np.array([(a, b) for k, (a, b) in enumerate(pair_indices) if not_bonds[k]])

    
    return atom_indices, pair_indices


def lookup_pairs_subset(all_pair_indices, subset_pair_indices, n_choose=None):
    """Convert pairs of atom indices into a list of indices

    Parameters
    ----------
    all_pair_indices : np.ndarray, dtype='int', shape=(N, 2)
        All allowed pairs of atom indices
    subset_pair_indices : np.ndarray, dtype=int, shape=(n, 2)
        A select subset of the atom pairs
    n_choose : int, default=None
        if not None, return at most this many indices

    Returns
    -------
    subset : np.ndarray, dtype=int, shape=(n)
        A numpy array with the integer indices that map subset_pair_indices
        onto all_pair_indices.  That is, subset[k] indices the value of 
        all_pair_indices that matches subset_pair_indices[k] 
    """


    n = all_pair_indices.max()
    
    all_keys = all_pair_indices[:, 0] + n * all_pair_indices[:, 1]
    optimal_keys = subset_pair_indices[:, 0] + n * subset_pair_indices[:, 1]
    subset = np.where(np.in1d(all_keys, optimal_keys))[0]
    if n_choose is not None:
        subset[0:min(len(subset), n_choose)] = subset[0:min(len(subset), n_choose)]

    return subset

    

class BaseSubsetFeaturizer(mixtape.featurizer.Featurizer):
    """Base class for featurizers that have a subset of active features.
    n_features refers to the number of active features.  n_max refers to the 
    number of possible features.
    
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
    """Subset featurizer based on atom pair distances."""
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

    def featurize(self, traj):
        if self.n_features > 0:
            features = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic) ** self.exponent
        else:
            features = np.zeros((traj.n_frames, 0))
        return features

    @property
    def pair_indices(self):
        return self.possible_pair_indices[self.subset]


class SubsetTrigFeaturizer(BaseSubsetFeaturizer):
    """Featurizer based on atom pair distances."""
    
    def featurize(self, traj):
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
        

class SubsetFeatureUnion(sklearn.pipeline.FeatureUnion):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None, subsets=None):
        self.subsets = subsets
        super(SubsetFeatureUnion, self).__init__(transformer_list, n_jobs=1, transformer_weights=None)
        

    def set_subsets(self, subsets):
        assert len(subsets) == len(self.transformer_list), "wrong len"
        for k, (_, featurizer) in enumerate(self.transformer_list):
            featurizer.subset = subsets[k]
        self.subsets = subsets

    @property
    def n_features(self):
        return sum([featurizer.n_features for (_, featurizer) in self.transformer_list])

    def 
