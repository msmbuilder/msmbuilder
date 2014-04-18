"""
Implementation of various useful utility functions for Hazan's algorithm.
"""

# Convenience functions; check if mdtraj has similar functionality
# already
def assert_list_of_types(Es, expected_type):
    """Checks whether input is a list of np.ndarray elements all of
       the same dimension dim

       Parameters
       __________
       Es: list
            Argument to check
       dim: int
            Expected dimension of Es
     """
    assert isinstance(Es, list)
    for i in range(len(Es)):
        Ei = Es[i]
        assert isinstance(Ei, expected_type)

def assert_list_of_square_arrays(Es, dim):
    """Checks whether input is a list of np.ndarray elements all of
       the same dimension dim

       Parameters
       __________
       Es: list
            Argument to check
       dim: int
            Expected dimension of Es
    """
    assert_list_of_types(Es, np.ndarray)
    for i in range(len(Es)):
       Ei = Es[i]
       assert np.shape(Ei) == (dim, dim)
