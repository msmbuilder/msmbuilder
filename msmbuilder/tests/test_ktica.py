from __future__ import absolute_import

import numpy as np
import itertools

from ..decomposition import tICA, KernelTICA
from ..example_datasets import fetch_alanine_dipeptide
from ..featurizer import AtomPairsFeaturizer


def test_1():

    bunch = fetch_alanine_dipeptide()

    atom_names = [a.element.symbol for a in bunch['trajectories'][0].top.atoms]
    heavy = [i for i in range(len(atom_names)) if atom_names[i] != 'H']
    atom_pairs = list(itertools.combinations(heavy, 2))

    featurizer = AtomPairsFeaturizer(atom_pairs)
    features = featurizer.transform(bunch['trajectories'][0:1])
    features = [features[0][::10]]

    tica = tICA(lag_time=1, n_components=2)
    ktica = KernelTICA(lag_time=1, kernel='linear', n_components=2)

    tica_out = tica.fit_transform(features)[0]
    ktica_out = ktica.fit_transform(features)[0]

    tica_out = tica_out * np.sign(tica_out[0])
    ktica_out = ktica_out * np.sign(ktica_out[0])

    # this isn't a very hard test to pass..
    diff = np.abs(tica_out - ktica_out) / tica_out.std(0)
    assert np.all(diff < 1)
