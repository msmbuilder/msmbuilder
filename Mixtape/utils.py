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
        kwargs = None  if file.endswith('.h5') else {'top': topology}
        t = md.load(file, **kwargs)
        t.superpose(topology, atom_indices=atom_indices)
        diff2 = (t.xyz[:, atom_indices] - topology.xyz[0, atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))

        X.append(x)
        i.append(np.arange(len(x)))
        f.extend([file]*len(x))

    return np.concatenate(X), np.concatenate(i), np.array(f)

