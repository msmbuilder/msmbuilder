.. _tutorial:
.. highlight:: bash

Tutorial
========

#.
   .. figure:: _static/fspeptide.png
       :align: right

       Fs Peptide

   The command-line tool is ``msmb``. Check that it's installed correctly and
   read the help options::

    msmb -h

#.
   In your own research, you probably have a large molecular dynamics
   dataset that you wish to analyze. For this tutorial, we will perform a
   quick analysis on a simple system: the Fs peptide.

#. We'll use MSMBuilder to create a set of sample python scripts to sketch
   out our project::

    msmb TemplateProject -h # read the possible options
    msmb TemplateProject

   This will generate a hierarchy of small scripts that will handle the
   boilerplate and organization inherent in MSM analysis.::

    .
    ├── analysis
    │   ├── dihedrals
    │   │   ├── tica
    │   │   │   ├── cluster
    │   │   │   │   ├── msm
    │   │   │   │   │   ├── msm-1-timescales-plot.py
    │   │   │   │   │   ├── msm-1-timescales.py
    │   │   │   │   │   ├── msm-2-microstate-plot.py
    │   │   │   │   │   ├── msm-2-microstate.py
    │   │   │   │   ├── cluster-plot.py
    │   │   │   │   ├── cluster.py
    │   │   │   │   ├── sample-clusters-plot.py
    │   │   │   │   ├── sample-clusters.py
    │   │   │   ├── ftrajs -> ../ftrajs
    │   │   │   ├── tica-plot.py
    │   │   │   ├── tica.py
    │   │   │   ├── tica-sample-coordinate-plot.py
    │   │   │   ├── tica-sample-coordinate.py
    │   │   ├── featurize-plot.py
    │   │   ├── featurize.py
    │   ├── gather-metadata-plot.py
    │   ├── gather-metadata.py
    ├── 0-test-install.py
    ├── 1-get-example-data.py
    └── README.md


   Each subsequent step in the MSM construction pipeline is a subdirectory.
   Try to understand this script. Run it to download the trajectory data.::

#. Retrieve the Fs peptide example data by running::

    python 1-get-example-data.py

   Ensure that you now have a directory named ``fs_peptide`` with 28 ``xtc``
   trajectories in it and a ``pdb`` topology file named ``fs-peptide.pdb``.
   Feel free to load one of these trajectories in VMD to get a sense of
   what they look like. Ensure that the symlinks ``top.pdb`` and ``trajs``
   resolve to the correct place. We use symlinks to map filenames specific
   to your project or protein into "standard" msmbuilder names. For example,
   if your set of trajectories was stored on another partition, you could
   use a symlink to point to the folder, name it ``trajs``, and the scripts
   will work without modification.

#. Begin our analysis::

    cd analysis/

#. We first generate a list of our trajectories and associated metadata.
   Examine the ``gather-metadata.py`` script. Note that it uses the ``xtc``
   filenames to generate an integer key for each trajectory. The script
   also extracts the length of each trajectory and stores the xtc filename.::

    python gather-metadata.py
    python gather-metadata-plot.py


   .. figure:: _static/lengths-hist.png
       :align: right

       Sometimes you'll have many different length-ed trajectories and
       this histogram will be interesting. All of our trajectories are 500ns
       though.

   The plot script contains several example functions of computing statistics
   on your dataset including aggregate length. It will also generate an ``html``
   rendering of the table of metadata. Exercise for the reader: modify the
   script to genereate ``png`` images intead of ``pdf`` vector graphics for
   the plots. Editor's note: use ``pdf`` for preperation of manuscripts
   because you can infinitely resize your plots.

#. We'll start reducing the dimensionality of our dataset by transforming
   the raw Cartesian coordinates into biophysical "features". We'll use
   dihedral angles. The templated project also includes subfolders ``landmarks``
   and ``rmsd`` for alternative approaches, but we'll ignore those for now.::

    cd dihedrals/

#. Examine the ``featurize.py`` script. Note that it loops through our trajectories
   using the convenience function ``itertrajs`` (which only ever holds one
   trajectory in RAM) and calls ``DihedralFeaturizer.partial_transform()``
   on each. Read more about :ref:`featurizers<featurization>` and MSMBuilder
   :ref:`API patterns<apipaterns>`. Run the scripts::

    python featurize.py
    python featurize-plot.py

   The plots will show you a box and whisker plot of each feature value. This
   is not very useful, but we wanted to make sure you can plot something
   for each step.

#. Dihedrals are too numerous to be interpretable. We can use :ref:`tica<decomposition>`
   to learn a small number of "kinetic coordinates" from our data.::

    cd tica/

#. Examine ``tica.py``. Note that it loads the feature trajectories, learns
   a model from them by calling ``fit()`` and then transforms the feature trajectories
   into "tica trajectories" by calling ``partial_transform()``
   on each (see :ref:`api patterns<apipatterns>`). The MSMBuilder API *does not*
   keep track of units. Our data was saved every 50 ps (Editor's note: this is
   way too frequent for a "real" simulation). The template script for learning
   our tica model sets the ``lag_time`` parameter to ``10``. This means 10 steps
   in our data. This translates to 500 ps in our case. Let's use something a little
   longer like 5 ns (= 100 steps). Edit the ``lag_time`` parameter to 100 and
   learn the model.::

    vim tica.py # edit lag_time -> 500
    python tica.py
    python tica-plot.py

#.
   .. figure:: _static/tica-heatmap.png
       :align: right

       tICA heatmaps provide a convenient 2d projection of your data
       onto which you can overlay more interesting info.

   The tICA plotting script makes a 2d histogram of our data. Note the apparent
   free energy well on the left of the figure. We might suspect that this is
   the folded state and the x-axis is an unfolding coordinate. We'll use
   this tica heatmap as a background for our further plots. tICA is extremely
   useful at taking hundreds of dihedral angles (for example) and distilling it
   into a handful of coordinates that we can plot.


