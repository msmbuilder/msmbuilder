# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.


import glob
import os
import re

import mdtraj as md
import pandas as pd


class _Parser(object):
    def parse_fn(self, fn):
        raise NotImplementedError


class GenericParser(_Parser):
    def __init__(self, fn_re=r'trajectory-([0-9]+)\.xtc', top_fn=""):
        self.fn_re = fn_re
        self.top_fn = top_fn

    def parse_fn(self, fn):
        ma = re.search(self.fn_re, fn)
        run = int(ma.group(1))
        meta = {
            'run': run,
            'traj_fn': fn,
            'top_fn': self.top_fn,
            'top_abs_fn': os.path.abspath(self.top_fn),
            'step_ps': 0,
        }
        with md.open(fn) as f:
            meta['nframes'] = len(f)
        return meta


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


def gather_metadata(fn_glob, parser):
    return pd.DataFrame(parser.parse_fn(fn) for fn in glob.iglob(fn_glob))
