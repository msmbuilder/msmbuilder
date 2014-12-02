from __future__ import print_function, absolute_import, division
import os
import functools
import warnings

# Copyright (C) 2012-2013 Marcus von Appen <marcus@sysfault.org>
#
# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.


class ExperimentalWarning(Warning):
    """Indicates that a certain class, function or behavior is in an
    experimental state.
    """
    def __init__(self, obj, msg=None):
        """Creates a ExperimentalWarning for the specified obj.

        If a message is passed in msg, it will be printed instead of the
        default message.
        """
        super(ExperimentalWarning, self).__init__()
        self.obj = obj
        self.msg = msg

    def __str__(self):
        if self.msg is None:
            line = "Warning: %s is in an experimental state." % repr(self.obj)
            return os.linesep.join(('', '"' * len(line), line, '"' * len(line)))
        return repr(self.msg)


def experimental(name=None):
    """A simple decorator to mark functions and methods as experimental."""
    def inner(func):
        @functools.wraps(func)
        def wrapper(*fargs, **kw):
            fname = name
            if name is None:
                fname = func.__name__
            warnings.warn("%s" % fname, category=ExperimentalWarning,
                          stacklevel=2)
            return func(*fargs, **kw)
        return wrapper
    return inner