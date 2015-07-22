.. _commandline:
.. highlight:: bash

Introductory Example (command line)
===================================

MSMBuilder is designed as a python library and a command-line program.  The
API can be much more powerful, and should be easy for researchers familiar
with Python. The command-line interface wraps the most common use cases.
Plotting and custom analysis will generally still be done in Python.


#. The command line tool is ``msmb``. Read the help options::

    msmb -h

#. In your own research, you probably have a large molecular dynamics
   dataset that you wish to analyze. For this tutorial, we will perform a
   quick analysis on a very simple system: alanine-dipeptide. Make a
   working directory and get an example dataset::

    mkdir ~/msmb_tutorial
    cd ~/msmb_tutorial
    msmb FsPeptide --data_home ./
    ls fs_peptide/

  You should see 28 ``.xtc`` trajectory files and an ``fs_peptide.pdb``
  topology file.

#. First, we need to turn the time-series of atomic coordinates we get from
   MD into useable "features". There are many choices of relevant features.
   We'll use phi and psi dihedral angles. Remember that ``\`` is the
   line-continuation operator in bash (for readability). You can enter this
   command on one line.::

    msmb DihedralFeaturizer  \
        --out featurizer.pkl \
        --transformed diheds \
        --top fs_peptide/fs_peptide.pdb \
        --trjs "fs_peptide/*.xtc"

   We give the topology file, a 'glob' expression to all of our
   trajectories, the atom indices we generated in the previous step, and an
   output filename.

   Read the help text (via ``msmb DihedralFeaturizer -h``) to make sure you
   understand each of these options.

#. We can train a kinetic model like tICA to transform our input features
   (dihedrals) into the most kinetically relevant linear combinations of
   those features::

    msmb tICA -i diheds/ \
        --out tica_model.pkl \
        --transformed tica_trajs.h5 \
        --n_components 4

  This code takes our feature trajectories from the previous step, fits a
  tICA model (saved as ``tica_model.pkl``) and transforms our feature
  trajectories into new trajectories (``tica_trajs.h5``). We specify that
  we want the top 4 slowest-components.

#. We can plot a 2d histogram of the top two tICA coordinates. Save the
   following to a file ``plot_tica.py``.

.. code-block:: python

    from msmbuilder.dataset import dataset
    from matplotlib import pyplot as plt
    import numpy as np

    trajs = dataset('tica_trajs.h5')    # Load file
    trajs = np.concatenate(trajs)       # Flatten list of trajectories
    plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
    plt.show()

#. Run it with python::

    python plot_tica.py

#. We can build an MSM from the tICA trajectories. First, we need to define
   states and assign our conformations to those states::

    msmb MiniBatchKMeans -i tica_trajs.h5 \
        --transformed labeled_trajs.h5 \
        --n_clusters 100

#. At this point, we have reduced the dimensionality of our problem from
   the thousands of atomic coordinates, to tens of dihedral angles, to a
   handful of tICA coordinates, to a 1D sequence of state labels. We
   construct an MSM from these labeled trajectories.::

    msmb MarkovStateModel -i labeled_trajs.h5 \
       --out msm.pkl
       --lag_time 1 \

#. Further analysis and plotting must be done in python.

.. code-block:: python

    from msmbuilder.utils import load
    msm = load("msm.pkl")

    print(msm.transmat_)
    print(msm.populations_)
    print(msm.timescales_)
    # ...


.. vim: tw=75
