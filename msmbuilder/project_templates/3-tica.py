from msmbuilder.dataset2 import load_trajs, save_trajs, save_generic
from msmbuilder.decomposition import tICA

tica = tICA(n_components=5, lag_time=10, kinetic_mapping=True)
meta, ftrajs = load_trajs("diheds")

tica.fit(ftrajs.values())

ttrajs = {}
for k, v in ftrajs.items():
    ttrajs[k] = tica.partial_transform(v)

save_trajs(ttrajs, 'ttrajs', meta)
save_generic(tica, 'tica.pickl')
