import numpy as np
from mixtape.cluster import RegularSpatial

def test_1():
    x = np.arange(10).reshape(10,1)

    model = RegularSpatial(d_min=0.99)
    model.fit([x])
    
    assert len(model.cluster_centers_) == 10
    np.testing.assert_array_equal(x, model.cluster_centers_)                                  
