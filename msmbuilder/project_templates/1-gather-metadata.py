import glob
import os
import re

import mdtraj as md
import pandas as pd


def parse_fn(fn):
    # Modify this function!
    # Include any and all metadata you want
    ma = re.search(r"trajectory-([0-9]+)\.xtc", fn)
    run = int(ma.group(1))
    meta = {
        'run': run,
        'traj_fn': fn,
        'top_fn': "{{topology_fn}}",
        'top_abs_fn': os.path.abspath("{{topology_fn}}"),
        'step_ps': {{timestep}},
    }
    with md.open(fn) as f:
        meta['nframes'] = len(f)
    return meta


meta = pd.DataFrame(parse_fn(fn) for fn in glob.glob("data/*.xtc"))
meta = meta.set_index('run').sort_index()
meta.to_pickle("meta.pandas.pickl")

print(meta.head())
print("...")
print(meta.tail())
print("Total trajectories:", len(meta))
