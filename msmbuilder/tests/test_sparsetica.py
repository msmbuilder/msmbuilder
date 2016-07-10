import numpy as np

from msmbuilder.decomposition import tICA, SparseTICA
from msmbuilder.example_datasets import DoubleWell


def build_dataset():
    slow = DoubleWell(random_state=0).get_cached().trajectories
    data = []

    # each trajectory is a double-well along the first dof,
    # and then 9 degrees of freedom of gaussian white noise.
    for s in slow:
        t = np.hstack((s, np.random.randn(len(s), 9)))
        data.append(t)
    return data


def test_doublewell():
    data = build_dataset()
    tica = tICA(n_components=1).fit(data)
    tic0 = tica.components_[0]

    stica = SparseTICA(n_components=1, verbose=False).fit(data)
    stic0 = stica.components_[0]

    np.testing.assert_array_almost_equal(stic0[1:], np.zeros(9))
    np.testing.assert_almost_equal(stic0[0], 0.58, decimal=1)
