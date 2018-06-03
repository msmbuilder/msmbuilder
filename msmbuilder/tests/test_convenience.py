
from msmbuilder.utils import unique

def test_unique():
    assert unique([1,2,3,3,2,1]) == [1,2,3]
    assert unique([3,3,2,2,1,1]) == [3,2,1]
    assert unique([3,1,2,1,2,3]) == [3,1,2]
