"""Find trajectories and associated metadata

{{header}}
"""

from msmbuilder.io import gather_metadata, save_meta, NumberedRunsParser

## Construct and save the dataframe
parser = NumberedRunsParser(
    traj_fmt="trajectory-{run}.xtc",
    top_fn="fs_peptide/fs-peptide.pdb",
    step_ps=50,
)
meta = gather_metadata("fs_peptide/*.xtc", parser)
save_meta(meta)
