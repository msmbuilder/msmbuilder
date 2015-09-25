.. _commandline:
.. highlight:: bash

Command Line Tutorial
=====================

MSMBuilder is designed as a python library and a command-line program.  The
API can be much more powerful, and should be easy for researchers familiar
with python to pick up. The command-line interface wraps the most common
use cases.

Tutorial
--------

1. The command line tool is ``msmb``. Read the help options::

    msmb -h

2. In your own research, you probably have a large molecular dynamics
   dataset that you wish to analyze. For this tutorial, we will perform a
   quick analysis on a very simple system: alanine-dipeptide. Make a
   working directory and get an example dataset::

    mkdir ~/msmb_tutorial
    cd ~/msmb_tutorial
    msmb AlanineDipeptide --data_home ./
    ls alanine_dipeptide/

  You should see 10 ``.dcd`` trajectory files and an ``ala2.pdb`` topology
  file.

3. First, we need to turn the time-series of atomic coordinates we get from
   MD into useable "features". There are many choices of relevant features.
   We'll use pairwise distances between heavy atoms. First, we need to
   select the heavy atoms. Remember that ``\`` is the line-continuation
   operator in bash (for readability). You can enter this command on one
   line.::

    msmb AtomIndices -d --heavy \
        -p alanine_dipeptide/ala2.pdb \
        -o AtomIndices.txt

   Read the help text (via ``msmb AtomIndices -h``) to make sure you
   understand each of these options.

4. Now we can compute the pairwise distances for each frame::

    msmb AtomPairsFeaturizer --transformed pair_features \
        --pair_indices AtomIndices.txt \
        --top alanine_dipeptide/ala2.pdb \
        --trjs "alanine_dipeptide/*.dcd"

   We give the topology file, a 'glob' expression to all of our
   trajectories, the atom indices we generated in the previous step, and an
   output filename.

5. We can train a kinetic model like tICA to transform our input features
   (atom pair distances) into the most kinetically relevant linear
   combinations of those features::

    msmb tICA -i pair_features \
        --out tica_model --transformed tica_trajs \
        --n_components 4

  This code takes our feature trajectories from the previous step, fits a
  tICA model (saved as ``tica_model.pkl``) and transforms our feature
  trajectories into new trajectories (``tica_trajs.h5``). We specify that
  we want the top 4 slowest-components.

6. We can plot a 2d histogram of the top two tICA coordinates. Save the
   following to a file ``plot_tica.py``.

.. code-block:: python

    from msmbuilder.dataset import dataset
    from matplotlib import pyplot as plt
    import numpy as np

    trajs = dataset('tica_trajs.h5')    # Load file
    trajs = np.concatenate(trajs)       # Flatten list of trajectories
    plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
    plt.show()

7. Run it with python::

    python plot_tica.py

