import numpy as np
from mixtape.markovstatemodel import _transition_counts

def test_1():
    with np.testing.assert_raises(ValueError):
        _transition_counts([1,2,3])

def test_2():
    c, m = _transition_counts([np.arange(10)])
    np.testing.assert_array_equal(c, np.eye(10, k=1))
    assert m.keys() == range(10)
    assert m.values() == range(10)
    


