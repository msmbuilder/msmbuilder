import os
import shutil
import warnings
import numpy as np
import pandas as pd


def backup(fn):
    if not os.path.exists(fn):
        return

    backnum = 1
    backfmt = "{fn}.bak.{backnum}"
    trial_fn = backfmt.format(fn=fn, backnum=backnum)
    while os.path.exists(trial_fn):
        backnum += 1
        trial_fn = backfmt.format(fn=fn, backnum=backnum)

    warnings.warn("{fn} exists. Moving it to {newfn}"
                  .format(fn=fn, newfn=trial_fn))
    shutil.move(fn, trial_fn)


def get_fn(base_fn, key):
    dfmt = "{}"
    ffmt = "{}.npy"
    if isinstance(key, tuple):  # TODO: check multiindex
        paths = [dfmt.format(k) for k in key[:-1]]
        paths += [ffmt.format(key[-1])]
        return os.path.join(base_fn, *paths)
    return os.path.join(base_fn, ffmt.format(key))


def save(meta, d, fn):
    backup(fn)
    os.mkdir(fn)
    for k in meta.index:
        v = d[k]
        np.save(get_fn(fn, k), v)


def load(meta, fn):
    if isinstance(meta, str):
        meta = pd.read_pickle(meta)
    d = {}
    for k in meta.index:
        d[k] = np.load(get_fn(fn, k))
    return meta, d
