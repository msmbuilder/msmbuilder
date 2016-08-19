from msmbuilder.io import NumberedRunsParser, HierarchyParser, GenericParser, \
    ParseWarning
import pandas as pd
import numpy as np
import warnings

filenames1 = [
    'traj0.xtc',
    'traj1.xtc',
    'traj2.xtc',
]

filenames2 = [
    'traj000.xtc',
    'traj001.xtc',
    'traj010.xtc',
]

filenames3 = [
    'PROJ9704/RUN0/CLONE10.xtc',
    'PROJ9704/RUN0/CLONE11.xtc',
    'PROJ9704/RUN1/CLONE10.xtc',
    'PROJ9704/RUN1/CLONE11.xtc',
    'PROJ9704.new/RUN1/CLONE12.xtc',
]


def gather_metadata(fn_list, parser):
    # Should be identical to msmbuilder.io.gather_metadata, except it takes
    # a list of filenames; not a glob expression
    meta = pd.DataFrame(parser.parse_fn(fn) for fn in fn_list)
    return meta.set_index(parser.index).sort_index()


def test_numbered_runs():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = NumberedRunsParser(traj_fmt="traj{run}.xtc")
        meta = gather_metadata(filenames1, nr)
    np.testing.assert_array_equal(meta.index, [0, 1, 2])


def test_padded_numbered_runs():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = NumberedRunsParser(traj_fmt="traj{run}.xtc")
        meta = gather_metadata(filenames2, nr)
    np.testing.assert_array_equal(meta.index, [0, 1, 10])


def test_numbered_runs_with_prefix():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = NumberedRunsParser(traj_fmt="traj{run}.xtc")
        meta = gather_metadata(["/test/directory/tree/{}".format(fn)
                                for fn in filenames1], nr)
    np.testing.assert_array_equal(meta.index, [0, 1, 2])


def test_numbed_runs_via_generic_parser():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = GenericParser(
            fn_re=r'traj(\d+)\.xtc',
            group_names=['test_run'],
            group_transforms=[int],
            top_fn="",
            step_ps=None,
        )
        meta = gather_metadata(filenames2, nr)
    np.testing.assert_array_equal(meta.index, [0, 1, 10])
    assert meta.index.name == 'test_run'


def test_hierarchy():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = HierarchyParser(n_levels=3)
        meta = gather_metadata(filenames3, nr)

    for x, y in zip(meta.index, [
        ('PROJ9704', 'RUN0', 'CLONE10.xtc'),
        ('PROJ9704', 'RUN0', 'CLONE11.xtc'),
        ('PROJ9704', 'RUN1', 'CLONE10.xtc'),
        ('PROJ9704', 'RUN1', 'CLONE11.xtc'),
        ('PROJ9704.new', 'RUN1', 'CLONE12.xtc'),
    ]):
        assert x == y, "{} != {}".format(x, y)

    assert meta.index.names == ('i0', 'i1', 'i2'), meta.index.names
    assert len(meta.loc['PROJ9704']) == 4
    assert len(meta.loc['PROJ9704', 'RUN0']) == 2

def test_hierarchy_with_names():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = HierarchyParser(levels=('proj', 'run', 'clone'))
        meta = gather_metadata(filenames3, nr)

    for x, y in zip(meta.index, [
        ('PROJ9704', 'RUN0', 'CLONE10.xtc'),
        ('PROJ9704', 'RUN0', 'CLONE11.xtc'),
        ('PROJ9704', 'RUN1', 'CLONE10.xtc'),
        ('PROJ9704', 'RUN1', 'CLONE11.xtc'),
        ('PROJ9704.new', 'RUN1', 'CLONE12.xtc'),
    ]):
        assert x == y, "{} != {}".format(x, y)

    assert meta.index.names == ('proj', 'run', 'clone'), meta.index.names
    assert len(meta.loc['PROJ9704']) == 4
    assert len(meta.loc['PROJ9704', 'RUN0']) == 2

def test_hierarchy_ignore_fext():
    with warnings.catch_warnings():
        # The files don't actually exist. We'll ignore those warnings
        warnings.simplefilter("ignore", ParseWarning)
        nr = HierarchyParser(levels=('proj', 'run', 'clone'), ignore_fext=True)
        meta = gather_metadata(filenames3, nr)

    for x, y in zip(meta.index, [
        ('PROJ9704', 'RUN0', 'CLONE10'),
        ('PROJ9704', 'RUN0', 'CLONE11'),
        ('PROJ9704', 'RUN1', 'CLONE10'),
        ('PROJ9704', 'RUN1', 'CLONE11'),
        ('PROJ9704', 'RUN1', 'CLONE12'),
    ]):
        assert x == y, "{} != {}".format(x, y)

    assert meta.index.names == ('proj', 'run', 'clone'), meta.index.names
    assert len(meta.loc['PROJ9704']) == 5
    assert len(meta.loc['PROJ9704', 'RUN0']) == 2
