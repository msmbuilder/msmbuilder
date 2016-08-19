"""Find trajectories and associated metadata

{{header}}

Meta
----
depends:
  - trajs
  - top.pdb
"""

from msmbuilder.io import gather_metadata, save_meta, NumberedRunsParser

## Construct and save the dataframe
parser = NumberedRunsParser(
    traj_fmt="trajectory-{run}.xtc",
    top_fn="top.pdb",
    step_ps=50,
)
meta = gather_metadata("trajs/*.xtc", parser)
save_meta(meta)
