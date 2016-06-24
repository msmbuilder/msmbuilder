import numpy as np


def continuous_gens(meta, levels_traj=None, levels_gen='gen'):
    """Make sure all gens are continuous

    "gen" is the Folding@Home terminology for parts of continuous
    trajectories. Problems happen, and sometimes one is missing or
    corrupted. This function finds problems and returns a cleaned up
    metadata dataframe

    This function makes some assumptions about the setup of your meta
    dataframe. Nameily, you should have a multiindex with levels named
    "proj", "run", "clone", "gen". The "gen" index should be numeric



    Parameters
    ----------
    meta : pd.DataFrame
        Dataframe with metadata about trajectory parts (gens).
    levels_traj : list of pd.MultiIndex names.
        Grouping by these multiindex levels should give one continuous
        trajectory per group. If None, defaults to ['proj', 'run', 'clone'].
    levels_gen : pd.MultiIndex name
        The multiindex level name that indexes trajectory parts. This
        index's values should be integers.

    Returns
    -------
    new_meta : pd.DataFrame
        DataFrame with non-contiguous gens removed
    problematic : list of dict
        Each entry gives the index (tuple of values corresponding
        to levels_traj) and last good gen index. Keyed by
        "index" and "last_good", resp.
    """
    if levels_traj is None:
        levels_traj = ['proj', 'run', 'clone']

    to_remove = []
    problematic = []
    # variable "prc" is a tuple of proj, run, clone
    # or whatever the user actually uses
    for prc, group in meta.groupby(level=levels_traj):
        gens = group.index.get_level_values(levels_gen)
        gens_should_be = np.arange(len(gens)) + gens.min()
        bad_positions = gens[gens != gens_should_be]
        if len(bad_positions > 0):
            problematic += [{'index': prc,
                             'last_good': bad_positions[0] - 1}]
            for bp in bad_positions:
                to_remove += [prc + (bp,)]
    new_meta = meta.drop(to_remove)
    return new_meta, problematic
