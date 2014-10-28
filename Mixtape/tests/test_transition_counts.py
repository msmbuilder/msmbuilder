import numpy as np
from six import PY3
from mixtape.markovstatemodel import _transition_counts


def test_1():
    # test that first argument must be a list of sequences
    with np.testing.assert_raises(ValueError):
        _transition_counts([1,2,3])


def test_2():
    # test a simple example
    c, m = _transition_counts([np.arange(10)])
    np.testing.assert_array_equal(c, np.eye(10, k=1))
    assert list(m.keys()) == list(range(10))
    assert list(m.values()) == list(range(10))


def test_3():
    # test the simple example with lag_time > 1
    c, m = _transition_counts([range(10)], lag_time=2)
    np.testing.assert_array_equal(c, np.eye(10, k=2))


def test_4():
    # try using strings as labels
    c, m = _transition_counts([['alpha', 'b', 'b', 'b', 'c']])
    np.testing.assert_array_equal(c, 1.0 * np.array([
        [0, 1, 0],
        [0, 2, 1],
        [0, 0, 0]
    ]))
    assert m == {'alpha': 0, 'b': 1, 'c': 2}


def test_5():
    # try using really big numbers, and we still want a small transition matrix
    c, m = _transition_counts([[100000000, 100000000, 100000001, 100000001]])
    np.testing.assert_array_equal(c, 1.0 * np.array([
        [1, 1],
        [0, 1],
    ]))
    assert m == {100000000: 0, 100000001: 1}

def test_6():
    c, m = _transition_counts([[0]])

def test_7():
    # deal with NaN, None?
    c, m = _transition_counts([[0, np.nan]])
    assert m == {0:0}
    np.testing.assert_array_equal(c, np.zeros((1,1)))

    c, m = _transition_counts([[np.nan]])
    assert m == {}
    np.testing.assert_array_equal(c, np.zeros((0,0)))

    if not PY3:
        c, m = _transition_counts([[None, None]])
        assert m == {}
        np.testing.assert_array_equal(c, np.zeros((0,0)))

