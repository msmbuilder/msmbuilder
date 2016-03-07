import numpy as np

from msmbuilder.msm import _strongly_connected_subgraph


def test_completely_disconnected_1():
    # what do you do with 1 state that is not even connected to itself?
    tC, m, p_r = _strongly_connected_subgraph(np.zeros((1, 1)))
    assert tC.shape == (0, 0)
    assert m == {}
    assert np.isnan(p_r)


def test_completely_disconnected_2():
    tC, m, p_r = _strongly_connected_subgraph(np.zeros((3, 3)))
    assert tC.shape == (0, 0)
    assert m == {}
    assert np.isnan(p_r)


def test_one_state():
    # but if that state does have a self-connection, it should be retained
    tC, m, p_r = _strongly_connected_subgraph(np.ones((1, 1)))
    assert tC.shape == (1, 1)
    assert m == {0: 0}
    np.testing.assert_almost_equal(p_r, 100)


def test_counts_1():
    C = np.array([[1, 0, 0],
                  [0, 1, 1],
                  [0, 1, 1]])

    tC, m, p_r = _strongly_connected_subgraph(np.array(C))
    np.testing.assert_array_equal(tC, np.array([[1, 1], [1, 1]]))
    assert m == {1: 0, 2: 1}
    np.testing.assert_almost_equal(p_r, 80.0)


def test_counts_2():
    C = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [0, 1, 1]])
    tC, m, p_r = _strongly_connected_subgraph(np.array(C))
    np.testing.assert_array_equal(tC, np.array([[1, 1], [1, 1]]))
    assert m == {1: 0, 2: 1}
    np.testing.assert_almost_equal(p_r, 83.333333333333)


def test_fully_connected():
    tC, m, p_r = _strongly_connected_subgraph(np.ones((3, 3)))
    np.testing.assert_array_almost_equal(tC, np.ones((3, 3)))
    assert m == {0: 0, 1: 1, 2: 2}
    np.testing.assert_almost_equal(p_r, 100.0)


def test_disconnected():
    tC, m, p_r = _strongly_connected_subgraph(np.eye(3))
    assert tC.shape == (1, 1)
    assert type(p_r) == np.float64


def test_upper_triangular():
    tC, m, p_r = _strongly_connected_subgraph(np.eye(3, k=1))
    assert tC.shape == (0, 0)
    assert m == {}
    np.testing.assert_almost_equal(p_r, 50.0)
