.. _tutorial:
.. highlight:: bash

Tutorial
========

#. The command-line tool is ``msmb``. Check that it's installed correctly and
   read the help options::

    msmb -h

#. In your own research, you probably have a large molecular dynamics
   dataset that you wish to analyze. For this tutorial, we will perform a
   quick analysis on a simple system: the FS peptide. 

   .. image:: _static/fspeptide.png
       :width: 50%
       :align: center

#. We'll use MSMBuilder to create a set of sample python scripts to sketch
   out our project::

    msmb ProjectTemplate -h # read the possible options
    msmb ProjectTemplate --steps 0

   This will write out scripts for "step 0" --- preparing our data.::

    vi 0-get-example-data.py

   Try to understand this script. Run it to download the trajectory data.::

    python 0-get-example-data.py

#. Now we'll do step 1 of an MSMBuilder project: organize our trajectories
   and cache their metadata.::

    msmb ProjectTemplate --steps 1
    vi 1-gather-metadata.py

   This file is a little complicated.
