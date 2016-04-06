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


class _Parser(object):
    def parse_fn(self, fn):
        raise NotImplementedError

    @property
    def index(self):
        raise NotImplementedError


class GenericParser(_Parser):
    def __init__(self,
                 fn_re,
                 transforms,
                 top_fn,
                 step_ps,
                 ):
        self.fn_re = re.compile(fn_re)
        self.top_fn = top_fn
        self.step_ps = step_ps
        self.transforms = transforms
        try:
            assert os.path.exists(top_fn)
        except:
            warnings.warn("Topology file doesn't actually exist! "
                          "You may (will) run into issues later when you "
                          "try to load it.")

    @property
    def index(self):
        return list(self.fn_re.groupindex.keys())


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
                          .format(fn, e))

        if self.step_ps is not None:
            meta['step_ps'] = self.step_ps

        # Get indices
        ma = self.fn_re.search(fn)
        if ma is None:
            raise ValueError("Filename {} did not match the "
                             "regular rexpression {}".format(fn, self.fn_re))
        meta.update({k: self.transforms[k](v)
                     for k, v in ma.groupdict().items()})
        return meta


class NumberedRunsParser(GenericParser):
    def __init__(self, traj_fmt="trajectory-{run}.xtc", top_fn="",
                 step_ps=None):
        # Test the input
        try:
            traj_fmt.format(run=0)
        except:
            raise ValueError("Invalid format string {}".format(traj_fmt))
        # Build a regex from format string
        s1, s2 = re.split(r'\{run\}', traj_fmt)
        capture_group = r'(?P<run>\d+)'
        fn_re = re.escape(s1) + capture_group + re.escape(s2)

        transforms = {'run': int}

        # Call generic
        super(NumberedRunsParser, self).__init__(fn_re, transforms,
                                                 top_fn, step_ps)


class HeirarchyParser(GenericParser):
    def __init__(self, levels=None, n_levels=None, top_fn="", step_ps=None):
        if (levels is None) == (n_levels is None):
            raise ValueError("Please specify levels or n_levels, but not both")

        if levels is None:
            levels = ["i{i}".format(i=i) for i in range(n_levels)]

        fn_re = r'\/'.join(r'(?P<{lvl}>[a-zA-Z0-9_\.\-]+)'.format(lvl=lvl)
                           for lvl in levels)

        transforms = {k: str for k in levels}
        super(HeirarchyParser, self).__init__(fn_re, transforms, top_fn,
                                              step_ps)


class GenericSplitParser(GenericParser):
    def __init__(self, fn_re=r'trajectory-([0-9]+)-([0-9]+)\.xtc', top_fn="",
                 step_ps=None):
        super(GenericSplitParser, self).__init__(fn_re=fn_re,
                                                 step_ps=step_ps,
                                                 top_fn=top_fn)

    def get_indices(self, fn):
        ma = re.search(self.fn_re, fn)
        run = int(ma.group(1))
        part = int(ma.group(2))
        return {
            'run': run,
            'part': part
        }

    @property
    def index(self):
        return ["run", 'part']


class FAHParser(_Parser):
    # FAH projects share a folder structure for project, run clone
    prc_re = re.compile(r"(PROJ(\d+))/RUN(\d+)/CLONE(\d+)")
    # But they differ in how they number gens. Try two regexs
    gen_re1 = re.compile(r"frame(\d+)\.xtc")
    gen_re2 = re.compile(r"results-(\d\d\d)/positions\.xtc")

    def __init__(self, core_type="21", top_fn=""):
        if core_type == 'a4':
            self.fn_re = None
        elif core_type in ['17', '18', '21']:
            self.fn_re = None
        else:
            raise ValueError("Unknown core type: {}".format(core_type))

        self.top_fn = top_fn

    def parse_fn(self, fn):
        ma_prc = self.prc_re.search(fn)
        ma_gen1 = self.gen_re1.search(fn)
        ma_gen2 = self.gen_re2.search(fn)
        if ma_gen1 is not None:
            overlapping_frame = True
            ma_gen = ma_gen1
        elif ma_gen2 is not None:
            overlapping_frame = False
            ma_gen = ma_gen2
        else:
            raise ValueError("Could not parse gen index")
        meta = {
            'projstr': ma_prc.group(1),
            'proj': int(ma_prc.group(2)),
            'run': int(ma_prc.group(3)),
            'clone': int(ma_prc.group(4)),
            'gen': int(ma_gen.group(1)),
            'traj_fn': fn,
            'top_fn': self.top_fn,
            'top_abs_fn': os.path.abspath(self.top_fn),
            'step_ps': 0,
            'overlapping_frame': overlapping_frame,
        }
        with md.open(fn) as f:
            meta['nframes'] = len(f)
        return meta

    @property
    def index(self):
        return ['proj', 'run', 'clone', 'gen']


def gather_metadata(fn_glob, parser):
    meta = pd.DataFrame(parser.parse_fn(fn) for fn in glob.iglob(fn_glob))
    return meta.set_index(parser.index).sort_index()
