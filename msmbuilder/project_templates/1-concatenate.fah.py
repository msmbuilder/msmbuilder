import numpy as np

from msmbuilder.dataset2 import load_meta

## Load
meta = load_meta()

## Consider overlapping frames
meta['true_nframes'] = meta['nframes'] - meta['overlapping_frame']

## Make sure we have continuous gens
to_remove = []
for (proj, run, clone), group in meta.groupby(level=['proj', 'run', 'clone']):
    gens = group.index.get_level_values('gen')
    gens_should_be = np.arange(len(gens)) + gens.min()
    bad_positions = gens[gens != gens_should_be]
    if len(bad_positions > 0):
        print("WARNING: Discontinuous gens in", proj, run, clone,
              "...Removing gens >=", bad_positions[0])
        for bp in bad_positions:
            to_remove += [(proj, run, clone, bp)]
meta = meta.drop(to_remove)
