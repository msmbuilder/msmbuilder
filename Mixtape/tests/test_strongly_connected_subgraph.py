import numpy as np
from mixtape.markovstatemodel import _strongly_connected_subgraph

def test_0():
    # what do you do with 1 state that is not even connected
    # to itself?
    tC, m = _strongly_connected_subgraph(np.zeros((1,1)))
    assert tC.shape == (0, 0)
    assert m == {}

def test_01():
    # but if that state does have a self-connection, it should be retained
    tC, m = _strongly_connected_subgraph(np.ones((1,1)))
    assert tC.shape == (1, 1)
    assert m == {0: 0}


def test_1():
    C = np.array([[1,0,0],
                  [0,1,1],
                  [0,1,1]])

    tC, m = _strongly_connected_subgraph(np.array(C))
    np.testing.assert_array_equal(tC, np.array([[1,1], [1,1]]))
    assert m == {1: 0, 2: 1}


def test_2():
    C = np.array([[1,1,0],
                  [0,1,1],
                  [0,1,1]])
    tC, m = _strongly_connected_subgraph(np.array(C))
    np.testing.assert_array_equal(tC, np.array([[1,1], [1,1]]))
    assert m == {1: 0, 2: 1}


def test_3():
    _strongly_connected_subgraph(np.zeros((3,3)))


def test_4():
    tC, m = _strongly_connected_subgraph(np.ones((3,3)))
    np.testing.assert_array_almost_equal(tC, np.ones((3,3)))
    assert m == {0:0, 1:1, 2:2}


def test_5():
    tC, m = _strongly_connected_subgraph(np.eye(3))
    assert tC.shape == (1,1)


def test_6():
    tC, m = _strongly_connected_subgraph(np.eye(3), -1)
    print(tC)

def test_7():
    tC, m = _strongly_connected_subgraph(np.eye(3, k=1))
    assert tC.shape == (0, 0)
    assert m == {}

