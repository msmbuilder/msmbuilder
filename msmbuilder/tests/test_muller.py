from msmbuilder.example_datasets import MullerPotential, load_muller, load_doublewell


def test_1():
    xx = load_muller(random_state=1)['trajectories']
    assert len(xx) == 10
    assert xx[0].ndim == 2
    assert xx[0].shape[1] == 2

    print(load_muller())
