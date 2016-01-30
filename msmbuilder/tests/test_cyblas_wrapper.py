# this file gets nose to find the tests that appear in the cython module
from msmbuilder.tests import test_cyblas


def test():
    count = 0
    for name in dir(test_cyblas):
        if name.startswith('test'):
            count += 1
            yield getattr(test_cyblas, name)
    if count == 0:
        assert False
