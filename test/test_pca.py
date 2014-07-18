import numpy as np
from sklearn.decomposition import PCA as PCAr

from mixtape.pca import PCA


def test_1():
    #Compare mixtape.pca with sklearn.decomposition

    np.random.seed(42)
    trajs = [np.random.randn(10, 3) for _ in range(5)]

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

