from msmbuilder.dataset2 import load
from msmbuilder.utils import dump
from msmbuilder.decomposition.interpretation import sample_dimension
import mdtraj as md

meta, ttrajs = load("meta.pandas.pickl", 'ttrajs')

inds = sample_dimension(ttrajs,
                        dimension=0,
                        n_frames=20, scheme='random')

dump(inds, "tica-dimension-0-inds.pickl")


def preload_top():
    top_fns = set(meta['top_fn'])
    assert len(top_fns) == 1, "You can only have 1 top when sampling"
    return md.load(top_fns.pop())


top = preload_top()

traj = None
for traj_i, frame_i in inds:
    frame = md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
    if traj is None:
        traj = frame
    else:
        traj += frame

traj.save("tica-dimension-0.xtc")
