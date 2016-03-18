
from msmbuilder.decomposition import tICA, ktICA
from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import AtomPairsFeaturizer
import numpy as np
import numpy.testing as npt
import itertools

def test_1():

    bunch = AlanineDipeptide().get()

    atom_names = [a.element.symbol for a in bunch['trajectories'][0].top.atoms]
    heavy = [i for i in xrange(len(atom_names)) if atom_names[i] != 'H']
    atom_pairs = list(itertools.combinations(heavy, 2))

    featurizer = AtomPairsFeaturizer(atom_pairs)
    features = featurizer.transform(bunch['trajectories'][0:1])
    features = [features[0][::10]]

    tica = tICA(lag_time=1, gamma=1.0 / features[0].shape[1], n_components=2)
    ktica = ktICA(lag_time=1, kernel='linear', eta=1.0 / (features[0].shape[0] * 2), n_components=2)

    tica_out = tica.fit_transform(features)[0]
    ktica_out = ktica.fit_transform(features)[0]

    tica_out = tica_out * np.sign(tica_out[0])
    ktica_out = ktica_out * np.sign(ktica_out[0])

    # this isn't a very hard test to pass..
    diff = np.abs(tica_out - ktica_out) / tica_out.std(0)
    assert np.all(diff < 1)


if __name__ == '__main__':
    test_1()
