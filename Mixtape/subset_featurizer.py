from contextlib import contextmanager
import itertools
import numpy as np
import mixtape.featurizer, mixtape.tica
import mdtraj as md

ATOM_NAMES = ["N", "CA", "CB", "C", "O", "H"]

def get_atompair_indices(trj0, keep_atoms=ATOM_NAMES, exclude_atoms=None, reject_bonded=True):
    """Get a list of acceptable atom pairs.

    Parameters
    ----------
    trj0 : mdtraj.Trajectory
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
    top, bonds = trj0.top.to_dataframe()
    
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

@contextmanager
def remember_subset(subset_featurizer):
    """Temporarily change the active subset of a featurizer and later restore it."""
    old_subset = subset_featurizer.subset.copy()
    yield
    subset_featurizer.set(old_subset)
    

class BaseSubsetFeaturizer(mixtape.featurizer.Featurizer):
    """Base class for featurizers that have a subset of active features.
    n_features refers to the number of active features.  n_max refers to the 
    number of possible features.
    
    """

    def __init__(self, trj0):
        self.trj0 = trj0

    @property
    def n_features(self):
        return len(self.subset)
        
    def propose(self):
        """Propose a feature to swap and a new one to replace it."""
        to_replace = np.random.choice(self.n_features)
        new_value = np.random.choice(self.n_max)
        old_value = self.subset[to_replace]
        
        return to_replace, new_value, old_value
    
    def set(self, subset):
        """Set the active features to a new subset."""
        self.subset = subset
    
    def propose_all(self, n_features):
        """Draw a random choice for all features."""
        return np.random.choice(self.n_max, n_features, replace=False)

    def featurize_single(self, trajectory, index):
        """Featurize a trajectory with only a single active subset (index).
        When proposing single feature swaps, this allows us to avoid 
        re-calculating the rest of the features.
        """
        
        with remember_subset(self):
            self.set(np.array([index]))
            X = self.featurize(trajectory)
        return X

class SubsetAtomPairs(BaseSubsetFeaturizer):
    """Subset featurizer based on atom pair distances."""
    def __init__(self, possible_pair_indices, reference_traj, periodic=False, alpha=1.0):
        
        self.possible_pair_indices = possible_pair_indices
        self.periodic = periodic
        self.alpha = alpha

    @property
    def n_max(self):
        return len(self.possible_pair_indices)

    def featurize(self, traj):
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
        return d ** self.alpha

    @property
    def pair_indices(self):
        return self.possible_pair_indices[self.subset]


class SubsetTrigFeaturizer(BaseSubsetFeaturizer):
    """Featurizer based on atom pair distances."""
    
    def featurize(self, traj):        
        dih = md.geometry.dihedral.compute_dihedrals(traj, self.which_atom_ind[self.subset])
        features = self.trig_function(dih)
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
        atom_indices, dih = md.geometry.dihedral.compute_phi(self.trj0)
        return atom_indices

class PsiMixin(object):
    @property
    def which_atom_ind(self):
        atom_indices, dih = md.geometry.dihedral.compute_psi(self.trj0)
        return atom_indices
    

class SubsetCosPhiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass

class SubsetCosPsiFeaturizer(SubsetTrigFeaturizer, CosMixin, PhiMixin):
    pass
    
class SubsetSinPhiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass
    
class SubsetSinPsiFeaturizer(SubsetTrigFeaturizer, SinMixin, PsiMixin):
    pass
        
        

class ListFeaturizer(BaseSubsetFeaturizer):
    """A Subset featurizer that concatenates a list of other featurizers,
    allowing selection from multiple classes of features.
    
    Parameters
    ----------
    n_features : int
        Number of active features
    featurizers : list(BaseSubsetFeaturizer)
        A list of subset featurizers to concatenate.
    
    To Do
    -----
    Allow weighting the classes of features.  Currently the proposals are
    uniform on the total number of possible features, meaning that
    featurizers with larger `n_max` will be sampled more thoroughly.
    """
    
    def __init__(self, n_features, featurizers):
        self.featurizers = featurizers

        self.index_mapping = np.array_split(np.arange(self.n_max), np.cumsum([f.n_max for f in self.featurizers])[:-1])
        self.cutoffs = np.array([mapping[0] for mapping in self.index_mapping])

        self.set(self.propose_all(n_features))

    @property
    def n_features(self):
        return sum([f.n_features for f in self.featurizers])

    @property
    def n_max(self):
        return sum([f.n_max for f in self.featurizers])

    def lookup_indices(self, indices):
        """Convert global indices to lists of states and local indices."""
        states = np.array([np.where(s >= self.cutoffs)[0][-1] for s in indices])
        local_states = indices - self.cutoffs[states]
        return states, local_states


    def featurize(self, traj):
        features = np.zeros((traj.n_frames, self.n_features))
        
        for k, featurizer in enumerate(self.featurizers):
            indices = np.where(self.states == k)[0]
            if len(indices) > 0:
                features[:, indices] = featurizer.featurize(traj)
        
        return features

        
    def set(self, subset):
        super(ListFeaturizer, self).set(subset)
        self.states, self.local_states = self.lookup_indices(self.subset)
        for k, featurizer in enumerate(self.featurizers):
            my_local_states = self.local_states[self.states == k]
            featurizer.set(my_local_states)
