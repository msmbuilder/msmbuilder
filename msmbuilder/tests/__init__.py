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

    m = {
        'argspec': 'inspect.getargspec() is deprecated'
    }

    if module == 'scipy._lib.decorator' and m['argspec'] in message:
        return

    if module == 'mdtraj.formats.hdf5' and m['argspec'] in message:
        return

    if module == 'statsmodels.base.wrapper' and m['argspec'] in message:
        return

    if module == 'nose.util' and m['argspec'] in message:
        return

    print("Warning: module:  ", module)
    print("Warning: message: ", message)
    return orig_warn(message=message, category=category,
                     stacklevel=stacklevel + 1)


warnings.warn = my_warn
