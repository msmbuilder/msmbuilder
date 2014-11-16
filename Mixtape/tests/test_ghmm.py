from __future__ import print_function, division

import numpy as np
from mixtape.hmm import GaussianFusionHMM
from mixtape.datasets import AlanineDipeptide
from mixtape.featurizer import SuperposeFeaturizer


def test_1():
    # creates a 4-state HMM on the ALA2 data. Nothing fancy, just makes
    # sure the code runs without erroring out
    dataset = AlanineDipeptide().get()
    trajectories = dataset.trajectories
    topology = trajectories[0].topology

    indices = topology.select('symbol C or symbol O or symbol N')
    featurizer = SuperposeFeaturizer(indices, trajectories[0][0])

    sequences = featurizer.transform(trajectories)
    hmm = GaussianFusionHMM(n_states=4, n_features=sequences[0].shape[1],
                            n_init=1)
    hmm.fit(sequences)

    assert len(hmm.timescales_ == 3)
    assert np.any(hmm.timescales_ > 50)
