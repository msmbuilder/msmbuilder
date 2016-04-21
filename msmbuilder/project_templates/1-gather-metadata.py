"""Find trajectories and associated metadata

{{header}}
"""

from msmbuilder.io import gather_metadata, GenericParser, save_meta

## Construct and save the dataframe
parser = GenericParser(r'trajectory-([0-9]+)\.xtc', 'fs-peptide.pdb')
meta = gather_metadata("data/*.xtc", parser)
save_meta(meta)

## Print a summary
print(meta.head())
print("Total trajectories:", len(meta))
