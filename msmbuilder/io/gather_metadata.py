# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.


import glob
import os
import re
import warnings

import mdtraj as md
import pandas as pd


class ParseWarning(UserWarning):
    pass


class _Parser(object):
    def parse_fn(self, fn):
        raise NotImplementedError

    @property
    def index(self):
        raise NotImplementedError


class GenericParser(_Parser):
    """Parse trajectories in a fully configurable manner

    Parameters
    ----------
    fn_re : str
        Regular expression with capture groups to transform trajectory
        filenames into keys
    group_names : list of str
        Capture group names (to serve as MultiIndex names)
    group_transforms : list of functions
        Apply these functions to capture groups
    top_fn : str
        Topology filename
    step_ps : int
        Timestep of frames in picoseconds

    """
    def __init__(self,
                 fn_re,
                 group_names,
                 group_transforms,
                 top_fn,
                 step_ps,
                 ):
        self.fn_re = re.compile(fn_re)
        self.group_names = group_names
        self.group_transforms = group_transforms
        self.top_fn = top_fn
        self.step_ps = step_ps
        try:
            assert os.path.exists(top_fn)
        except:
            warnings.warn("Topology file doesn't actually exist! "
                          "You may (will) run into issues later when you "
                          "try to load it.", ParseWarning)

        assert len(group_names) == len(group_transforms)
        assert len(group_names) == self.fn_re.groups

    @property
    def index(self):
        return list(self.group_names)

    def parse_fn(self, fn):
        meta = {
            'traj_fn': fn,
            'top_fn': self.top_fn,
            'top_abs_fn': os.path.abspath(self.top_fn),
        }
        try:
            with md.open(fn) as f:
                meta['nframes'] = len(f)
        except Exception as e:
            warnings.warn("Could not determine the number of frames for {}: {}"
                          .format(fn, e), ParseWarning)

        if self.step_ps is not None:
            meta['step_ps'] = self.step_ps

        # Get indices
        ma = self.fn_re.search(fn)
        if ma is None:
            raise ValueError("Filename {} did not match the "
                             "regular rexpression {}".format(fn, self.fn_re))
        meta.update({gn: transform(ma.group(gi))
                     for gn, transform, gi
                     in zip(self.group_names, self.group_transforms,
                            range(1, len(self.group_names) + 1))
                     })
        return meta


class NumberedRunsParser(GenericParser):
    """Parse trajectories that are numbered with integers.

    Parameters
    ----------
    traj_fmt : str
        A format string with {run} in it that gives the filename to look
        for. {run} will be captured and turned into an integer.
    top_fn : str
        Topology filename
    step_ps : int
        Trajectory frame step in picoseconds

    """

    def __init__(self, traj_fmt="trajectory-{run}.xtc", top_fn="",
                 step_ps=None):
        # Test the input
        try:
            traj_fmt.format(run=0)
        except:
            raise ValueError("Invalid format string {}".format(traj_fmt))
        # Build a regex from format string
        s1, s2 = re.split(r'\{run\}', traj_fmt)
        capture_group = r'(\d+)'
        fn_re = re.escape(s1) + capture_group + re.escape(s2)

        # Call generic
        super(NumberedRunsParser, self).__init__(
            fn_re=fn_re,
            group_names=['run'],
            group_transforms=[int],
            top_fn=top_fn,
            step_ps=step_ps
        )


class HierarchyParser(GenericParser):
    """Parse a hierarchical index from files nested in directories

    A trajectory with path:
      PROJ9704.new/RUN4/CLONE10.xtc
    will be given an index of
      ('PROJ9704.new', 'RUN4', 'CLONE10.xtc')
    If you set the flag ignore_fext=True, it will be given an index of
      ('PROJ9704', 'RUN4', 'CLONE10')

    Parameters
    ----------
    levels : list of str
        Level names
    n_levels : int
        Number of levels. Either this or levels must be provided (but
        not both). The levels will be named i0, i1, ... i(n-1)
    top_fn : str
        Topology filename
    step_ps : int
        Number of picoseconds per frame
    ignore_fext : bool
        Ignore file extensions. If set to true, this will fail if there is
        more than one "." per file/directory name. Anything after a "." is
        considered a file extension and will be ignored; including in
        directory names
    """

    def __init__(self, levels=None, n_levels=None, top_fn="", step_ps=None,
                 ignore_fext=False):
        if (levels is None) == (n_levels is None):
            raise ValueError("Please specify levels or n_levels, but not both")

        if levels is None:
            levels = ["i{i}".format(i=i) for i in range(n_levels)]

        if ignore_fext:
            # regex notes:
            #  1. (?:...) means non-capturing group
            subre = r'([a-zA-Z0-9_\-]+)(?:\.[a-zA-Z0-9]+)?'
        else:
            subre = r'([a-zA-Z0-9_\.\-]+)'

        fn_re = r'\/'.join(subre for _ in levels)

        super(HierarchyParser, self).__init__(
            fn_re=fn_re,
            group_names=levels,
            group_transforms=[str for _ in levels],
            top_fn=top_fn,
            step_ps=step_ps
        )


def gather_metadata(fn_glob, parser):
    """Given a glob and a parser object, create a metadata dataframe.

    Parameters
    ----------
    fn_glob : str
        Glob string to find trajectory files.
    parser : descendant of _Parser
        Object that handles conversion of filenames to metadata rows.
    """
    meta = pd.DataFrame(parser.parse_fn(fn) for fn in glob.iglob(fn_glob))
    return meta.set_index(parser.index).sort_index()
