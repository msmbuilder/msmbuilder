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

    if module == 'scipy._lib.decorator' and "inspect.getargspec() is deprecated" in message:
        return

    if module == 'mdtraj.formats.hdf5' and "inspect.getargspec() is deprecated" in message:
        return

    return orig_warn(message=message, category=category,
                     stacklevel=stacklevel + 1)


warnings.warn = my_warn
