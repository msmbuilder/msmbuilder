import numpy as np
from six import PY3

from msmbuilder.msm import _transition_counts


def test_argument():
    # test that first argument must be a list of sequences
    with np.testing.assert_raises(ValueError):
        _transition_counts([1, 2, 3])


def test_upper_triangular():
    # test a simple example
    c, m = _transition_counts([np.arange(10)])
    np.testing.assert_array_equal(c, np.eye(10, k=1))
    assert list(m.keys()) == list(range(10))
    assert list(m.values()) == list(range(10))


def test_lag_time():
    # test the simple example with lag_time > 1
    c, m = _transition_counts([range(10)], lag_time=2)
    np.testing.assert_array_equal(c, 0.5 * np.eye(10, k=2))


def test_string_labels():
    # try using strings as labels
    c, m = _transition_counts([['alpha', 'b', 'b', 'b', 'c']])
    np.testing.assert_array_equal(c, 1.0 * np.array([
        [0, 1, 0],
        [0, 2, 1],
        [0, 0, 0]
    ]))
    assert m == {'alpha': 0, 'b': 1, 'c': 2}


def test_big_counts():
    # try using really big numbers, and we still want a small transition matrix
    c, m = _transition_counts([[100000000, 100000000, 100000001, 100000001]])
    np.testing.assert_array_equal(c, 1.0 * np.array([
        [1, 1],
        [0, 1],
    ]))
    assert m == {100000000: 0, 100000001: 1}


def test_no_counts():
    c, m = _transition_counts([[0]])


def test_nan_and_none():
    # deal with NaN, None?
    c, m = _transition_counts([[0, np.nan]])
    assert m == {0: 0}
    np.testing.assert_array_equal(c, np.zeros((1, 1)))

    c, m = _transition_counts([[np.nan]])
    assert m == {}
    np.testing.assert_array_equal(c, np.zeros((0, 0)))

    if not PY3:
        c, m = _transition_counts([[None, None]])
        assert m == {}
        np.testing.assert_array_equal(c, np.zeros((0, 0)))


def test_lag_time_norm():
    X = np.arange(6)
    C, _ = _transition_counts([X], lag_time=3)
    np.testing.assert_array_almost_equal(C, np.eye(6, k=3) / 3)


def test_sliding_window():
    X = np.arange(10)
    C1, m1 = _transition_counts([X], lag_time=3, sliding_window=False)
    C2, m2 = _transition_counts([X[::3]], sliding_window=True)
    np.testing.assert_array_almost_equal(C1, C2)
    assert m1 == m2
