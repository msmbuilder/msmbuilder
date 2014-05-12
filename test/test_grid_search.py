import os
import time
import numpy as np
from sklearn.grid_search import GridSearchCV
from mixtape.grid_search import DistributedGridSearchCV
from sklearn.svm import SVC


def setup_module():
    os.system('ipcluster start --daemonize')
    time.sleep(1)


def teardown_module():
    os.system('ipcluster stop')


def test_1():
    X = np.random.randn(100, 2)
    y = np.random.randint(2, size=100)

    grid1 = DistributedGridSearchCV(SVC(), param_grid={'C': range(1,10)})
    grid1.fit(X, y)
    g1 = grid1.grid_scores_

    grid2 = GridSearchCV(SVC(), param_grid={'C': range(1,10)})
    grid2.fit(X, y)
    g2 = grid2.grid_scores_

    for a, b in zip(g1, g2):
        np.testing.assert_array_almost_equal(
            a['cv_validation_scores'], b.cv_validation_scores)
