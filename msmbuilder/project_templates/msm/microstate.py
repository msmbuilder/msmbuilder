"""Make a microstate MSM

{{header}}
"""

from msmbuilder.io import load_trajs, save_trajs, save_generic
from msmbuilder.msm import MarkovStateModel

## Load
meta, ktrajs = load_trajs('ktrajs')

## Fit
msm = MarkovStateModel(lag_time=2, n_timescales=10, verbose=False)
msm.fit(list(ktrajs.values()))

## Transform
microktrajs = {}
for k, v in ktrajs.items():
    microktrajs[k] = msm.partial_transform(v)

## Save
print(msm.summarize())
save_generic(msm, 'msm.pickl')
save_trajs(microktrajs, 'microktrajs', meta)
