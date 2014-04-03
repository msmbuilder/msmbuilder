import numpy as np
import pickle
import tables
import mdtraj as md
from sklearn.externals.joblib import dump, load
from mixtape.ghmm import GaussianFusionHMM
from sklearn.pipeline import Pipeline
from sklearn import base


class MDTrajFeaturizer(base.BaseEstimator):
    def __init__(self, conformation, chunk=1000, stride=1):
        self.chunk = chunk
        self.stride = stride
        self.conformation = conformation
        
    def iterload(self, filename):
        kwargs = {} if filename.endswith('.h5') else {'top': conformation}
        for t in md.iterload(filename, chunk=self.chunk, stride=self.stride, **kwargs):
            yield t

    def fit(self, filenames, Y):
        pass

    def transform(self, filenames):
        data = []
        for fn in filenames:
            X = []
            for chunk in self.iterload(fn):
                X.append(self.featurize(chunk))
            data.append(np.concatenate(X))
        return data

    def fit_transform(self, filenames, y=None):
        return self.transform(filenames)


class SuperposeFeaturizer(MDTrajFeaturizer):
    def __init__(self, atom_indices, conformation, chunk=1000, stride=1):
        self.atom_indices = atom_indices
        self.conformation = conformation
        self.n_features = len(self.atom_indices)
        super(SuperposeFeaturizer, self).__init__(conformation, chunk, stride)

    def featurize(self, traj):
        traj.superpose(self.conformation, atom_indices=self.atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] -
                 self.conformation.xyz[0, self.atom_indices]) ** 2
        x = np.sqrt(np.sum(diff2, axis=2))
        return x


class LagtimeSubsampler(base.BaseEstimator):
    def __init__(self, lag_time):
        self.lag_time = lag_time

    def fit(self, dataset, y=None):
        pass

    def transform(self, dataset):
        dataset = [X[::self.lag_time] for X in dataset]
        return dataset

    def fit_transform(self, dataset, y=None):
        return self.transform(dataset)



pipeline = Pipeline([
    ('featurizer', SuperposeFeaturizer(np.arange(20), md.load('/Users/rmcgibbo/datasets/Fip35-WW/structures/ww_native.pdb'))),
    ('subsampler', LagtimeSubsampler(lag_time=2)),
    ('hmm', GaussianFusionHMM(n_states=4, n_features=20))
])


pipeline.fit(['trj0.h5'])

from mdtraj.utils import timing
with timing():
    dump(pipeline, 'out', compress=9)

pipe2 = load('out')
print pipe2

