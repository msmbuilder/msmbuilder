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


class Featurizer(object):
    def __init__(self):
        pass

    def featurize(self, filenames, atom_indices, topology):
        pass

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
