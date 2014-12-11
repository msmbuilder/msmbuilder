import numpy as np
from msmbuilder.example_datasets import DoubleWell
from msmbuilder.decomposition import tICA, SparseTICA
from msmbuilder.decomposition.speigh import imported_cvxpy
from numpy.testing.decorators import skipif


def build_dataset():
    slow = DoubleWell(random_state=0).get()['trajectories']
    data = []

    # each trajectory is a double-well along the first dof,
    # and then 9 degrees of freedom of gaussian white noise.
    for s in slow:
        t = np.hstack((s, np.random.randn(len(s), 9)))
        data.append(t)
    return data

@skipif(not imported_cvxpy, 'cvxpy is required')
def test_1():
    data = build_dataset()
    tica = tICA(n_components=1).fit(data)
    tic0 = tica.components_[0]
    print('tICA\n', tic0)

    stica = SparseTICA(n_components=1, verbose=True).fit(data)
    stic0 = stica.components_[0]
    print('Sparse tICA\n', stic0)
    assert np.allclose(stic0, [1,0,0,0,0,0,0,0,0,0])
