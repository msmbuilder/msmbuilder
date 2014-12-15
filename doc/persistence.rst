Datasets and Persistence
========================

MSMBuilder loads and saves two types of objects to and from the filesystem:
datasets and models.

Datasets
--------
In MSMBuilder, a dataset is a collection of timeseries, or "sequences".
Conceptually, each timeseries usually represents a single molecular dynamics
trajectory, and may be represented in a number of different formats

 - A sequence may be an instance of ``mdtraj.Trajectory``, a molecular dynamics
   trajectory object.
 - A sequence may be a ``double`` or ``float``-valued 2D array with shape
   ``n_frames x n_features``, representing the projection of each frame in
   molecular dynamics trajectory into some vector space of dimension :math:`\mathbb{R}^{n_{features}}`. The leading dimension with length
   ``n_frames`` indexes over the timeseries.

   ..

        Example: :class:`AtomPairsFeaturizer` transforms trajectories into
        a vector space with dimension specified by the number of pairs of
        atoms supplied.

 - A sequence may be an integer-valued 1D array with shape ``n_frames``.

   ..

       Example: When a dataset is clustered, the transformed result of a
       trajectory is a sequence of the cluster index of each frame.


Datasets on Disk
~~~~~~~~~~~~~~~~

Trajectories (read-only)
"""""""""""""""""""""""""
Trajectory datasets are loaded using MDTraj. This requires
specifying a `glob <http://en.wikipedia.org/wiki/Glob_%28programming%29>`_
pattern for the trajectories, as well as the topology. MSMBuilder does not write
trajectory datasets.


    Example::

      >>> from msmbuilder.dataset import dataset
      >>> ds = dataset('trajectories/*.xtc', top='topology.pdb')
      >>> for traj in ds:
      ...     # iterate over trajectories
      ...     print(traj)
 

HDF5 Array Datasets (read or write)
"""""""""""""""""""""""""""""""""""

Array datasets can be read and written in two formats: ``hdf5`` and ``npy-dir``.

With HDF5, the dataset containing all of the trajectories is contained in a
single file on disk. This is generally the most convenient, but can be unwieldy
for datasets larger than 10GB. The transformed output of the ``msmb tICA``,
``msmb PCA``, and all of the clustering commands are stored in HDF5 format.

npy-dir Array Datasets (read or write)
""""""""""""""""""""""""""""""""""""""

The ``npy-dir`` format stores the dataset in a directory on disk, containing
each sequence in a separate uncompressed file. This is the most suitable for
large datasets, because it enables features like memory-mapped IO. The transformed output of ``msmb *Featurizer`` commands are stored in
``npy-dir`` format.

Provenance Information
""""""""""""""""""""""
When msmbuilder saves a dataset, it also saves information which can be used to
trace the provenance of the dataset.

  Example::

    $ msmb AtomPairsFeaturizer --out atom_pairs  --trjs '*.dcd'  --pair_indices atom_indices.txt  --top top.pdb
    AtomPairsFeaturizer(exponent=1.0,
              pair_indices=array([[ 1,  4],
           [ 1,  5],
           ...,
           [15, 18],
           [16, 18]]),
              periodic=False)
    100%|########################################################################################|Time: 0:00:00

    Saving transformed dataset to 'atom_pairs/'
    To load this dataset interactive inside an IPython
    shell or notebook, run

      $ ipython
      >>> from msmbuilder.dataset import dataset
      >>> ds = dataset('atom_pairs/')

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


Model persistence
-----------------

MSMBuilder models can be losslessly persisted to disk using Python's pickle
infrastructure. We recommend using the functions :func:`msmbuilder.utils.load`
and :func:`msmbuilder.utils.dump` to load and save models respectively.
