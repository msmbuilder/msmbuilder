from warnings import warn as orig_warn
import warnings
import sys


def my_warn(message, category=None, stacklevel=1):
    # taken from warnings module
    # Get context information
    try:
        caller = sys._getframe(stacklevel)
    except ValueError:
        globals = sys.__dict__
        lineno = 1
    else:
        globals = caller.f_globals
        lineno = caller.f_lineno
    module = globals['__name__']
    filename = globals.get('__file__')

    # Note: I went through each test (alphabetically) and added ignore rules.
    # Each time I needed a new skip rule, I labeled the section with which
    # file caused me to include it. Since these rules apply to all files,
    # it would be incorrect to assume that each rule is *only* for the file
    # in its section header.

    # test_agglomerative

    if module == 'scipy._lib.decorator' and "inspect.getargspec() is deprecated" in message:
        return

    if module == 'mdtraj.formats.hdf5' and "inspect.getargspec() is deprecated" in message:
        return

    # test_alphaanglefeaturizer

    return orig_warn(message=message, category=category,
                     stacklevel=stacklevel + 1)


warnings.warn = my_warn
