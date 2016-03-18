
import numpy as np
from numpy.testing import assert_approx_equal
from mdtraj.testing import eq
from msmbuilder.decomposition.ktica import ktICA
from msmbuilder.decomposition.kernal_approx import Nystroem
from msmbuilder.decomposition.tica import tICA


def test_1():
    # make sure it can fit transform
    np.random.seed(42)
    X = np.random.randn(10, 3)

    ktica = ktICA(k_name='rbf', k_n_components=5, k_gamma=None, t_n_components=1,
                  t_lag_time=1, t_gamma=0.05, t_weighted_transform=False)
    y2 = ktica.fit_transform([np.copy(X)])[0]
    eq(y2.shape,(10,1))

def test_nystoerm_step():
    X = np.random.randn(100, 10)
    ktica = ktICA(k_name='rbf', k_n_components=3, k_gamma=None, t_n_components=1,
                  t_lag_time=1, t_gamma=0.05, t_weighted_transform=False)
    ktica.nystroem.set_params(random_state=0)
    nys = Nystroem(kernel='rbf', n_components=3, gamma=None, random_state=0)
    y1 = ktica.nystroem.fit_transform([X])
    y2 = nys.fit_transform([X])
    assert np.allclose(y1, y2)


def test_tica_step():
    X = np.random.randn(100, 5)
    tica = tICA(lag_time=1, n_components=1, gamma=0)
    ktica = ktICA(k_name='rbf', k_n_components=3, k_gamma=None, t_n_components=1,
                  t_lag_time=1, t_gamma=0.0)
    tica.fit([X])
    ktica.tica.fit([X])
    y1 = ktica.tica.transform([X])
    y2 = tica.transform([X])
    assert np.allclose(y1, y2)


def test_pipeline():
    X = np.random.randn(100, 5)
    ktica = ktICA(k_name='rbf', k_n_components=3, k_gamma=None, t_n_components=1,
                  t_lag_time=1, t_gamma=0.0)
    ktica.nystroem.set_params(random_state=0)
    y1 = ktica.fit_transform([X])

    nys = Nystroem(kernel='rbf', n_components=3, gamma=None, random_state=0)
    tica = tICA(lag_time=1, n_components=1, gamma=0)

    y2_1 = nys.fit_transform([X])
    y2_2 = tica.fit_transform(y2_1)

    assert np.allclose(y1, y2_2)


def test_shape():
    model = ktICA(k_name='rbf', k_n_components=5, k_gamma=None, t_n_components=3,
                  t_lag_time=1, t_gamma=0.05, t_weighted_transform=False)
    model.fit([np.random.randn(100, 10)])
    eq(model.nystroem.components_.shape, (5, 10))
    eq(model.tica.n_features, 5)
    eq(model.tica.eigenvalues_.shape, (3,))
    eq(model.tica.eigenvectors_.shape, (5, 3))
    eq(model.tica.components_.shape, (3, 5))
