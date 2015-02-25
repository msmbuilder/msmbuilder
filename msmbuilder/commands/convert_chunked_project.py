# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""Convert an MD dataset with chunked trajectories into a standard format.

This script will walk down the filesystem, starting in ``root``, looking
for directories which contain one or more files matching ``pattern`` using
shell-style "glob" formatting. In each of these directories, the matching
files will be sorted by filename (natural order), interpreted as chunks
of a single contiguous MD trajectory, and loaded.

[This script assumes that trajectory files in the same leaf directory
are chunks of a contiguous MD trajectory. If that's not the case for your
dataset, this is the WRONG script for you.]

The concatenated trajectory will be saved to disk inside the ``outdir``
directory, under a filename set by the ``outfmt`` format string.

A record of conversion will be saved inside the directory as a JSON Lines file
[http://jsonlines.org/], which contains a newline-delimited collection of
JSON records, each of which is of the form
    {"chunks": ["path/to/input-chunk"], "filename": "output-file"}
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import json
import traceback
from datetime import datetime
from fnmatch import fnmatch

import mdtraj as md
from mdtraj.utils import timing
from mdtraj.formats.registry import _FormatRegistry
EXTENSIONS = _FormatRegistry.loaders.keys()

try:
    from scandir import walk as _walk
except ImportError:
    from os import walk as _walk

from ..cmdline import Command, argument, argument_group, FlagAction
from ..dataset import _keynat


class ConvertChunkedProject(Command):
    _group = '0-Support'
    _concrete = True
    description = __doc__
    root = argument('root', help='''Root of the directory structure containing
        the MD trajectories to convert (filesystem path)''')
    outdir = argument('outdir', help='''Output directory in which to save
        converted trajectories.''', default='trajectories')
    req = argument_group('required arguments')
    pattern = req.add_argument('--pattern', help='''Glob pattern for matching
        trajectory chunks (example: \'frame*.xtc\'). Use single quotes to
        specify expandable patterns''', required=True)
    top = req.add_argument('-t', '--topology', help='''Path to system topology
        file (.pdb / .prmtop / .psf)''',
        type=md.core.trajectory._parse_topology, required=True)
    outfmt = argument('--outfmt', help='''Format for output trajectories. This
        should be a python string format specifier, which is parameterized by a
        single int. The filename extension can specify any supported MDTraj
        trajectory format ({}).'''.format(', '.join(EXTENSIONS)),
        default='traj-%08d.dcd')
    discard = argument('--discard-first', help='''Flag to discard the initial
        frame in each chunk before concatenating trajectories. This is
        necessary for some old-style Folding@Home datasets''',
        action=FlagAction, default=False)
    stride =argument('--stride', type=int, help='''Convert every stride-th
        frame from the trajectories''', default=1)
    dry = argument('--dry-run', help='''Trace the execution, without
        actually running any actions''', action='store_true')

    def __init__(self, args):
        self.args = args
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        try:
            args.outfmt % 1
        except TypeError:
            self.error('"%s" is not a valid string format. It should contain '
                       'a single %%d specifier' % args.outfmt)

    def start(self):
        args = self.args
        metadata_path = os.path.join(args.outdir, 'trajectories.jsonl')

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = [json.loads(line) for line in f]
        else:
            metadata = []

        print_datetime()
        for chunk_fns in walk_project(args.root, args.pattern):
            if chunk_fns in {tuple(e['chunks']) for e in metadata}:
                print('Skipping %s. Already processed' % os.path.dirname(chunk_fns[0]))
                continue

            try:
                with timing('Loading %s: %d files' % (os.path.dirname(chunk_fns[0]), len(chunk_fns))):
                    traj = load_chunks(chunk_fns, args.topology,
                                       args.stride,
                                       discard_first=args.discard_first)
            except (ValueError, RuntimeError):
                print('======= Error loading chunks! Skipping ==========', file=sys.stderr)
                print('-' * 60)
                traceback.print_exc(file=sys.stderr)
                print('-' * 60)
                continue

            out_filename = args.outfmt % len(metadata)
            assert out_filename not in {tuple(e['filename']) for e in metadata}
            assert not os.path.exists(os.path.join(args.outdir, out_filename))

            print('Saving... ', end=' ')
            if not args.dry_run:
                traj.save(os.path.join(args.outdir, out_filename))
            print('%s:  [%s]' % (out_filename, ', '.join(os.path.basename(e)
                                                         for e in chunk_fns)))

            metadata_item = {'filename': out_filename, 'chunks': chunk_fns}
            metadata.append(metadata_item)

            # sync it back to disk
            if not args.dry_run:
                with open(metadata_path, 'a') as f:
                    json.dump(metadata_item, f)
                    f.write('\n')

        print_datetime()
        print('Finished successfully!')


def walk_project(root, pattern):
    for dirpath, dirnames, filenames in _walk(root):
        filenames = sorted([os.path.join(dirpath, fn)
                            for fn in filenames
                            if fnmatch(fn, pattern)], key=_keynat)
        if len(filenames) > 0:
            yield tuple(filenames)


def load_chunks(chunk_fns, top, stride, discard_first=True):
    trajectories = []
    for fn in chunk_fns:
        t = md.load(fn, stride=stride, top=top)
        if discard_first:
            t = t[1:]
        trajectories.append(t)
    out = trajectories[0]
    if len(trajectories) > 1:
        out = out.join(trajectories[1:])
    return out


def print_datetime(file=sys.stdout):
    print('Currently:        %s' % datetime.now().strftime('%b %d %Y %H:%M:%S'),
          file=sys.stdout)
