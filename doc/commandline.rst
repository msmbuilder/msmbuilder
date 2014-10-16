.. _commandline:

Command Line Interface (CLI)
============================

Mixtape is really designed (at this point) as a python library. It's much more
powerful that way, and not any less convenient (especially when you're deailing
with large datasets). But there is a limited CLI as well.

Tutorial
--------

1. The command line tool is `msmb`. Read the help options::

    msmb -h

2. In your own research, you probably have a large molecular dynamics dataset
   that you wish to analyze. For this tutorial, we will perform a quick
   analysis on a very simple system: alanine-dipeptide.Make a working directory
   and get an example dataset::

    mkdir ~/msmb_tutorial
    cd ~/msmb_tutorial
    msmb alanine

3. First, we need to turn the time-series of atomic coordinates we get from
   MD into useable "features". There are many choices of relevant features.
   We'll use pairwise distances between heavy atoms. First, we need to select
   the heavy atoms::

    msmb select_pairs --which heavy

4. Now we can compute the pairwise distances for each frame::

    msmb atompairs --trajs "data/trajs/*.h5"

5. Choose a lagtime::

    msmb implied_timescales -lb 1 -le 20 -i 2


