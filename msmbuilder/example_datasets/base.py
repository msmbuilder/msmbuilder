import shutil
import time
from os import environ
from os.path import join
from os.path import exists
from os.path import expanduser
from os import makedirs
from functools import wraps
from six.moves.urllib.error import HTTPError
import sys


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


def has_msmb_data():
    """We provide a conda package containing the saved data.

    This package was introduced because the figshare downloads could
    be 'iffy' at times.

    Returns
    -------
    has_msmb_data : bool
        Whether we think the package is installed
    path : str
        The path (if it exists). otherwise None
    """
    msmb_data_dir = join(sys.prefix, 'share', 'msmb_data')
    if exists(msmb_data_dir):
        return True, msmb_data_dir
    else:
        return False, None


def _expand_and_makedir(data_home):
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def get_data_home(data_home=None):
    """Return the path of the msmbuilder data dir.

    As of msmbuilder v3.6, this function will prefer data downloaded via
    the msmb_data conda package (and located within the python installation
    directory). If this package exists, we will use its data directory as
    the data home. Otherwise, we use the old logic:

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'msmbuilder_data'
    in the user's home folder.

    Alternatively, it can be set by the 'MSMBUILDER_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is not None:
        return _expand_and_makedir(data_home)

    msmb_data, msmb_data_path = has_msmb_data()
    if msmb_data:
        return _expand_and_makedir(msmb_data_path)

    data_home = environ.get('MSMBUILDER_DATA', join('~', 'msmbuilder_data'))
    return _expand_and_makedir(data_home)


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
