import cPickle
import json
import numpy as np
import mdtraj as md


def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass

def featurize_all(filenames, featurizer, topology):
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


def load_superpose_timeseries(filenames, atom_indices, topology):
    X = []
    i = []
    f = []
    for file in filenames:
        kwargs = {}  if file.endswith('.h5') else {'top': topology}
        t = md.load(file, **kwargs)
        t.superpose(topology, atom_indices=atom_indices)
        diff2 = (t.xyz[:, atom_indices] - topology.xyz[0, atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))

        X.append(x)
        i.append(np.arange(len(x)))
        f.extend([file]*len(x))

    return np.concatenate(X), np.concatenate(i), np.array(f)


def load(filename):
    print(filename)
    featurizer = cPickle.load(open(filename))
    return featurizer


class Featurizer(object):
    def __init__(self):
        pass

    def featurize(self, traj):
        pass

    def save(self, filename):
        cPickle.dump(featurizer, open(args.filename, 'w'))


class SuperposeFeaturizer(object):
    def __init__(self, atom_indices, reference_traj):
        self.atom_indices = atom_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.atom_indices)
        
    def featurize(self, traj):

        traj.superpose(self.reference_traj, atom_indices=self.atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] - self.reference_traj.xyz[0, self.atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))
        
        return x

class AtomPairsFeaturizer(object):
    def __init__(self, pair_indices, reference_traj, periodic=False):
        self.pair_indices = pair_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.pair_indices)
        self.periodic = periodic
        
    def featurize(self, traj):
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
        return d
