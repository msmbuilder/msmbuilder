.. _io:
.. currentmodule:: msmbuilder.io

I/O
===

.. autosummary::
    :toctree: _io/

    load_trajs
    save_trajs
    load_meta
    save_meta

Expected columns
----------------

step_ps
    The timestep of each trajectory in picoseconds.

FAH
~~~

proj, run, clone
    Folding@Home trajectories are indexed by a tuple of
    project, run, clone (by default). The ``meta`` file should have
    these as the index (use ``meta.set_levels(['proj', 'run', 'clone'])``.
    Some FAH utilities expect these names. These functions take argument
    ``levels_traj``
gen
    Sometimes you put a gen. Functions take an argument ``levels_gen``.

