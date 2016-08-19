.. _datasets:

Datasets
========

MSMBuilder prior to version 3.6 relied on the ``dataset`` utility for
trajectory loading and intermediate data persistence.
The two types of objects to be persisted on disk are datasets and models.

Datasets
--------

A ``dataset`` is a collection of timeseries, or "sequences".
Each timeseries usually represents a single molecular
dynamics trajectory, and may be represented in a number of different
formats

- A sequence may be an instance of ``mdtraj.Trajectory``, a molecular
  dynamics trajectory object.

- A sequence may be a ``numpy`` 2D array with shape ``n_frames x
  n_features``, representing the projection of each frame in molecular
  dynamics trajectory into some vector space of dimension
  :math:`\mathbb{R}^{n_{features}}`. The leading dimension of length
  ``n_frames`` indexes over the timeseries. For example,
  :ref:`featurization<featurization>` takes a list of trajectories and returns a
  list of feature arrays.

- A sequence may be an integer-valued 1D array with shape ``n_frames``.
  For example, :ref:`clustering<cluster>` takes a list of feature arrays and
  returns a list of sequences of state indices.


Datasets on Disk
~~~~~~~~~~~~~~~~

MSMBuilder can read and write datasets to and from disk in two
formats: ``hdf5`` and ``dir-npy``. From the Python API, you must choose
which format to write. The command-line application chooses the most
sensible option for you.

With HDF5, the dataset containing all of the trajectories is contained in a
single file on disk. This is generally the most convenient, but can be
unwieldy for large datasets. The transformed output of ``msmb tICA``,
``msmb PCA``, and clustering commands is stored in HDF5 format.

The ``dir-npy`` format stores the dataset as a collection of uncompressed
numpy ``.npy`` files in a directory on disk. This is the most suitable for
large datasets, because it enables features like memory-mapped IO. The
transformed output of ``msmb`` Featurizer commands are stored in
``dir-npy`` format.


Trajectory Datasets - Read only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trajectory datasets are loaded using MDTraj. This requires specifying a
`glob <http://en.wikipedia.org/wiki/Glob_%28programming%29>`_ pattern for
the trajectories, as well as the topology. MSMBuilder does not write
trajectory datasets.
 

Provenance Information
~~~~~~~~~~~~~~~~~~~~~~

When msmbuilder saves a dataset, it also saves information which can be
used to trace the provenance of the dataset.

.. code-block:: bash

    $ msmb AtomPairsFeaturizer --out atom_pairs  --trjs '*.dcd'  --pair_indices atom_indices.txt  --top top.pdb

    [...]

    $ ls atom_pairs
    00000000.npy   00000002.npy   00000004.npy   00000006.npy   00000008.npy   PROVENANCE.txt
    00000001.npy   00000003.npy   00000005.npy   00000007.npy   00000009.npy

    $ cat atom_pairs/PROVENANCE.txt
    MSMBuilder Dataset:
      MSMBuilder:	3.0.0-beta.dev-99bc8a9
      Command:	msmb AtomPairsFeaturizer --out atom_pairs  --trjs '*.dcd'  --pair_indices
      Path:		atom_pairs/
      Username:	rmcgibbo
      Hostname:	Computer-3.local
      Date:		December 01, 2014 12:16 AM
      Comments:

    == Derived from ==
    MDTraj dataset:
      path:		*.dcd
      topology:	/Users/rmcgibbo/msmbuilder_data/alanine_dipeptide/top.pdb
      stride:	1
      atom_indices	None


Models
------

MSMBuilder models can be losslessly persisted to disk using Python's pickle
infrastructure. We recommend using the functions
:func:`msmbuilder.utils.load` and :func:`msmbuilder.utils.dump` to load and
save models respectively. The pickle format is not secure against malicious
attacks. Don't load MSMBuilder models from untrusted sources.


Functions
---------

.. autofunction:: msmbuilder.dataset.dataset
.. autofunction:: msmbuilder.utils.dump
.. autofunction:: msmbuilder.utils.load

.. vim: tw=75

