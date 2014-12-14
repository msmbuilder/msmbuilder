import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as PCAr
from msmbuilder.decomposition import PCA
from msmbuilder.cluster import KCenters
random = np.random.RandomState(42)
trajs = [random.randn(10, 3) for _ in range(5)]

def test_1():
    #Compare msmbuilder.pca with sklearn.decomposition

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


def test_2():
    # Tet that PCA it works in a msmbuilder pipeline

    p = Pipeline([('pca', PCA()), ('cluster', KCenters())])
    p.fit(trajs)
