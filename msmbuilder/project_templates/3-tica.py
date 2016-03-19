from msmbuilder.dataset2 import save, load
from msmbuilder.utils import dump
from msmbuilder.decomposition import tICA

tica = tICA(n_components=5, lag_time=10, kinetic_mapping=True)
meta, vmtrajs = load("meta.pandas.pickl", "vmtrajs")

tica.fit(vmtrajs.values())

ttrajs = {}
for k, v in vmtrajs.items():
    ttrajs[k] = tica.partial_transform(v)

save(meta, ttrajs, "ttrajs")
dump(tica, 'tica.pickl')
