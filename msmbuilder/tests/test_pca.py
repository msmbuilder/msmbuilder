import numpy as np
from sklearn.decomposition import PCA as PCAr
from sklearn.pipeline import Pipeline

from msmbuilder.cluster import KCenters
from msmbuilder.decomposition import PCA

random = np.random.RandomState(42)
trajs = [random.randn(10, 3) for _ in range(5)]


def test_vs_sklearn():
    # Compare msmbuilder.pca with sklearn.decomposition

    pcar = PCAr()
    pcar.fit(np.concatenate(trajs))

    pca = PCA()
    pca.fit(trajs)

    y_ref1 = pcar.transform(trajs[0])
    y1 = pca.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)
    np.testing.assert_array_almost_equal(pca.components_, pcar.components_)
    np.testing.assert_array_almost_equal(pca.explained_variance_,
                                         pcar.explained_variance_)
    np.testing.assert_array_almost_equal(pca.mean_, pcar.mean_)
    np.testing.assert_array_almost_equal(pca.n_components_, pcar.n_components_)
    np.testing.assert_array_almost_equal(pca.noise_variance_,
                                         pcar.noise_variance_)


def test_pipeline():
    # Test that PCA it works in a msmbuilder pipeline

    p = Pipeline([('pca', PCA()), ('cluster', KCenters())])
    p.fit(trajs)


def test_generator():
    # Check to see if it works with a generator

    traj_dict = dict((i, t) for i, t in enumerate(trajs))

    pcar = PCAr()
    pcar.fit(np.concatenate(trajs))

    pca = PCA()
    # on python 3, dict.values() returns a generator
    pca.fit(traj_dict.values())

    y_ref1 = pcar.transform(trajs[0])
    y1 = pca.transform(trajs)[0]

    np.testing.assert_array_almost_equal(y_ref1, y1)
    np.testing.assert_array_almost_equal(pca.components_, pcar.components_)
    np.testing.assert_array_almost_equal(pca.explained_variance_,
                                         pcar.explained_variance_)
    np.testing.assert_array_almost_equal(pca.mean_, pcar.mean_)
    np.testing.assert_array_almost_equal(pca.n_components_, pcar.n_components_)
    np.testing.assert_array_almost_equal(pca.noise_variance_,
                                         pcar.noise_variance_)
