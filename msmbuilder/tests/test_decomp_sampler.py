import numpy as np

from msmbuilder.decomposition import tICA
from msmbuilder.decomposition.interpretation import sample_dimension, sample_region


def test_sample_dimension():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    data = [X, X, X]

    tica = tICA(n_components=2, lag_time=1)

    t_d = tica.fit_transform(data)

    res = sample_dimension(t_d, 0, 10, scheme="linear")

    res2 = sample_dimension(t_d, 1, 10, scheme="linear")

    assert len(res)==len(res2)==10


def test_sample_region():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    data = [X, X, X, X]

    tica = tICA(n_components=2, lag_time=1)

    t_d = tica.fit_transform(data)

    pt_dict = {0:0.1, 1:0.2}
    res = sample_region(t_d,pt_dict, 2)
    assert len(res)==2


