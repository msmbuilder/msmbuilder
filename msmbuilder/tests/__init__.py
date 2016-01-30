import sys
import warnings
from warnings import warn as orig_warn


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

    m = {
        'argspec': 'inspect.getargspec() is deprecated'
    }

    # test_agglomerative

    if module == 'scipy._lib.decorator' and m['argspec'] in message:
        return

    if module == 'mdtraj.formats.hdf5' and m['argspec'] in message:
        return

    # test_alphaanglefeaturizer

    if module == 'statsmodels.base.wrapper' and m['argspec'] in message:
        return

    # test_bayes_ratematrix
    # test_clustering
    # test_commands

    if module == 'nose.util' and m['argspec'] in message:
        return

    print("Warning: module:  ", module)
    print("Warning: message: ", message)
    return orig_warn(message=message, category=category,
                     stacklevel=stacklevel + 1)

    # test_cyblas
    # test_cyblas_wrapper
    # test_dataset
    # test_estimator_subclassing
    # test_featureunion
    # test_featurizer


warnings.warn = my_warn
