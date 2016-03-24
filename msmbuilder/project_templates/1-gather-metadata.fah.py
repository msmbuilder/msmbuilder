import glob
import os
import re

import mdtraj as md
import pandas as pd

from msmbuilder.io import save_meta

## Specify where to find our projects
xa4_glob = 'RUN*/CLONE*/frame*.xtc'
x21_glob = 'RUN*/CLONE*/results-???/positions.xtc'

projects = {
    'PROJ9704': x21_glob,
    'PROJ9752': xa4_glob,
}


## Set up parsing of fah-style filenames
def parse_fn(fn):
    ma_prc = re.search(r"(PROJ(\d+))/RUN(\d+)/CLONE(\d+)", fn)
    ma_gen1 = re.search(r"frame(\d+)\.xtc", fn)
    ma_gen2 = re.search(r"results-(\d\d\d)/positions\.xtc", fn)
    if ma_gen1 is not None:
        overlapping_frame = True
        ma_gen = ma_gen1
    elif ma_gen2 is not None:
        overlapping_frame = False
        ma_gen = ma_gen2
    else:
        raise ValueError()
    meta = {
        'projstr': ma_prc.group(1),
        'proj': int(ma_prc.group(2)),
        'run': int(ma_prc.group(3)),
        'clone': int(ma_prc.group(4)),
        'gen': int(ma_gen.group(1)),
        'traj_fn': fn,
        'top_fn': "{{topology_fn}}",
        'top_abs_fn': os.path.abspath("{{topology_fn}}"),
        'step_ps': {{timestep}},
        'overlapping_frame': overlapping_frame,
    }
    with md.open(fn) as f:
        meta['nframes'] = len(f)
    return meta


## Find multiple projects
def chain_glob():
    for proj, globstr in projects.items():
        yield from glob.iglob("{proj}/{globstr}"
                              .format(proj=proj, globstr=globstr))


## Construct and save the dataframe
meta = pd.DataFrame(parse_fn(fn) for fn in chain_glob())
meta = meta.set_index(['proj', 'run', 'clone', 'gen']).sort_index()
save_meta(meta)

## Print a summary
print(meta.head())
print("...")
print(meta.tail())
print("Total trajectory fragments:", len(meta))
