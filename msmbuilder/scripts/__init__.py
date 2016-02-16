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

    if 'inspect.getargspec() is deprecated' in message:
        return

    # This explicit check is necessary for python  < 3.5 maybe??
    if category is None:
        category = UserWarning

    return orig_warn(message=message, category=category,
                     stacklevel=stacklevel + 1)


warnings.warn = my_warn
