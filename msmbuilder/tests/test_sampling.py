import numpy as np

from msmbuilder.decomposition import tICA
from msmbuilder.io.sampling import sample_dimension


def test_sample_dimension():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    data = [X, X, X]

    tica = tICA(n_components=2, lag_time=1).fit(data)
    tica_trajs = {k: tica.partial_transform(v) for k, v in enumerate(data)}
    res = sample_dimension(tica_trajs, 0, 10, scheme="linear")
    res2 = sample_dimension(tica_trajs, 1, 10, scheme="linear")

    assert len(res) == len(res2) == 10
