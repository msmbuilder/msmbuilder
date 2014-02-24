import cPickle
import numpy as np
import mdtraj as md

def featurize_all(filenames, featurizer, topology):
    """Iterate over filenames, load trajectories, and featurize."""
    X = []
    i = []
    f = []
    for file in filenames:
        kwargs = {}  if file.endswith('.h5') else {'top': topology}
        t = md.load(file, **kwargs)
        x = featurizer.featurize(t)

        X.append(x)
        i.append(np.arange(len(x)))
        f.extend([file]*len(x))

    return np.concatenate(X), np.concatenate(i), np.array(f)


def load(filename):
    """Load a featurizer from a cPickle file."""
    featurizer = cPickle.load(open(filename))
    return featurizer


class Featurizer(object):
    """Base class for Featurizer objects."""
    def __init__(self):
        pass

    def featurize(self, traj):
        pass

    def save(self, filename):
        cPickle.dump(self, open(filename, 'w'))


class SuperposeFeaturizer(Featurizer):
    """Featurizer based on euclidian atom distances to reference structure."""
    def __init__(self, atom_indices, reference_traj):
        self.atom_indices = atom_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.atom_indices)
        
    def featurize(self, traj):

        traj.superpose(self.reference_traj, atom_indices=self.atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] - self.reference_traj.xyz[0, self.atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))
        
        return x

class AtomPairsFeaturizer(Featurizer):
    """Featurizer based on atom pair distances."""
    def __init__(self, pair_indices, reference_traj, periodic=False):
        self.pair_indices = pair_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.pair_indices)
        self.periodic = periodic
        
    def featurize(self, traj):
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
        return d
