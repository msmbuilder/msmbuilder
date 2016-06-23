from msmb_data import load_muller, MullerPotential
from msmbuilder.utils import array2d


def test_func():
    xx = load_muller(random_state=1)['trajectories']
    assert len(xx) == 10
    assert xx[0].ndim == 2
    assert xx[0].shape[1] == 2
    array2d(xx)


def test_class():
    xx = MullerPotential(random_state=123122).get()['trajectories']
    assert len(xx) == 10
    assert xx[0].ndim == 2
    assert xx[0].shape[1] == 2
    array2d(xx)
