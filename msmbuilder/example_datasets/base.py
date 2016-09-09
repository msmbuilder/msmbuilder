import numbers
import shutil
import sys
import time
from functools import wraps
from io import BytesIO
from os import environ
from os import makedirs
from os.path import exists
from os.path import expanduser
from os.path import join
from zipfile import ZipFile

import numpy as np
from six.moves.urllib.error import HTTPError
from six.moves.urllib.request import urlopen
from sklearn.utils import check_random_state


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

    def get_cached(self):
        raise NotImplementedError


class _MDDataset(Dataset):
    target_directory = ""  # set in subclass
    data_url = ""  # set in subclass

    def __init__(self, data_home=None, verbose=True):
        self.data_home = get_data_home(data_home)
        self.data_dir = join(self.data_home, self.target_directory)
        self.cached = False
        self.verbose = verbose

    def _msmbdata_cache(self):
        if self.verbose:
            print("Copying {} from msmb_data package to {}"
                  .format(self.target_directory, self.data_home))
        msmb_data = has_msmb_data()
        assert msmb_data is not None
        shutil.copytree("{}/{}".format(msmb_data, self.target_directory),
                        self.data_dir)

    @retry(3)
    def _figshare_cache(self):
        if self.verbose:
            print('downloading {} from {} to {}'
                  .format(self.target_directory, self.data_url,
                          self.data_home))
        fhandle = urlopen(self.data_url)
        buf = BytesIO(fhandle.read())
        zip_file = ZipFile(buf)
        makedirs(self.data_dir)
        for name in zip_file.namelist():
            zip_file.extract(name, path=self.data_dir)

    @retry(3)
    def cache(self):
        if not exists(self.data_home):
            makedirs(self.data_home)

        if not exists(self.data_dir):
            if has_msmb_data() is not None:
                self._msmbdata_cache()
            else:
                self._figshare_cache()
        elif self.verbose:
            print("{} already is cached".format(self.target_directory))

        self.cached = True

    def get_cached(self):
        raise NotImplementedError

    def get(self):
        if not self.cached:
            self.cache()
        return self.get_cached()


class _NWell(Dataset):
    """Base class for brownian dynamics on a potential

    Parameters
    ----------
    data_home : optional, default: None
        Specify another cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    random_state : {int, None}, default: None
        Seed the psuedorandom number generator to generate trajectories. If
        seed is None, the global numpy PRNG is used. If random_state is an
        int, the simulations will be cached in ``data_home``, or loaded from
        ``data_home`` if simulations with that seed have been performed already.
        With random_state=None, new simulations will be performed and the
        trajectories will not be cached.
    """

    target_name = ""  # define in subclass
    n_trajectories = 0  # define in subclass
    version = 1  # override in subclass if parameters are updated

    def __init__(self, data_home=None, random_state=None):
        self.data_home = get_data_home(data_home)
        self.data_dir = join(self.data_home, self.target_name)
        self.random_state = random_state
        self.cache_path = self._get_cache_path(random_state)

    def _get_cache_path(self, random_state):
        path = "{}/version-{}/randomstate-{}".format(self.data_dir,
                                                     self.version,
                                                     self.random_state)
        return path

    def _load(self, path):
        return [np.load("{}/{}.npy".format(path, i))
                for i in range(self.n_trajectories)]

    def _save(self, path, trajectories):
        assert len(trajectories) == self.n_trajectories
        if not exists(path):
            makedirs(path)
        for i, traj in enumerate(trajectories):
            np.save("{}/{}.npy".format(path, i), traj)

    def cache(self):
        random = check_random_state(self.random_state)
        if not exists(self.data_dir):
            makedirs(self.data_dir)

        if self.random_state is None:
            trajectories = self.simulate_func(random)
            return trajectories

        if not isinstance(self.random_state, numbers.Integral):
            raise TypeError('random_state must be an int')
        if exists(self.cache_path):
            return self._load(self.cache_path)

        trajectories = self.simulate_func(random)
        self._save(self.cache_path, trajectories)
        return trajectories

    def get_cached(self):
        if self.cache_path is None:
            raise ValueError("You must specify a random state to get "
                             "cached trajectories.")
        trajectories = self._load(self.cache_path)
        return Bunch(trajectories=trajectories, DESCR=self.description())

    def get(self):
        trajectories = self.cache()
        return Bunch(trajectories=trajectories, DESCR=self.description())

    def simulate_func(self, random):
        # Implement in subclass
        raise NotImplementedError

    def potential(self, x):
        # Implement in subclass
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
    path : str or None
        The path (if it exists). otherwise None
    """
    msmb_data_dir = join(sys.prefix, 'share', 'msmb_data')
    if exists(msmb_data_dir):
        return msmb_data_dir
    else:
        return None


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

    msmb_data = has_msmb_data()
    if msmb_data is not None:
        return _expand_and_makedir(msmb_data)

    data_home = environ.get('MSMBUILDER_DATA', join('~', 'msmbuilder_data'))
    return _expand_and_makedir(data_home)


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)
