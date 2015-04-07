"""Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
# 2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# 2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
# Adapted for msmbuilder from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/base.py

import shutil
from os import environ
from os.path import join
from os.path import exists
from os.path import expanduser
from os import makedirs
from functools import wraps
from six.moves.urllib.error import HTTPError


class Dataset(object):
    @classmethod
    def description(cls):
        """Get a description from the Notes section of the docstring."""
        lines = [s.strip() for s in cls.__doc__.splitlines()]
        note_i = lines.index("Notes")
        return "\n".join(lines[note_i + 2:])

    def cache(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def get_data_home(data_home=None):
    """Return the path of the msmbuilder data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'msmbuilder_data'
    in the user home folder.

    Alternatively, it can be set by the 'MSMBUILDER_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('MSMBUILDER_DATA', join('~', 'msmbuilder_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def retry(max_retries=1):
    """ Retry a function `max_retries` times. """
    def retry_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            while num_retries <= max_retries:
                try:
                    ret = func(*args, **kwargs)
                    break
                except HTTPError:
                    if num_retries == max_retries:
                        raise
                    num_retries += 1
                    time.sleep(5)
            return ret
        return wrapper
    return retry_func
