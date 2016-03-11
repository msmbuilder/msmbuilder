.. _cluster:
.. currentmodule:: msmbuilder.cluster

Clustering
==========

Clustering MD trajectories groups the the data [#f1]_ into a set of
clusters such that conformations in the same cluster are structurally
similar to one another, and conformations in different clusters are
structurally distinct. The questions that arise are

#. How should "structurally similar" be defined? What distance metric
   should be used?

#. Given the distance metric, what algorithm should be used to actually
   cluster the data?

On point 1, there is no consensus in the protein MD literature. Popular
distance metrics include cartesian root-mean squared deviation of atomic
positions (RMSD) [#f3]_, distances based on the number of native contacts
formed, distances based on the difference in backbone dihedral angles, and
probably others.

On point 2, "Optimal" clustering is NP-hard [#f2]_, so there's usually a
tradeoff between clustering quality and computational cost. For that reason
MSMBuilder has a variety of different clustering algorithms implemented.

Algorithms
----------

All clustering algorithms in MSMBuilder follow the following basic API.
Hyperparameters, including the number of clusters, random seeds, the
distance metric (if applicable) are passed to the class constructor.
Then, the computation is done by calling ``fit(sequences)``. The argument
to ``fit`` should be a *list* of molecular dynamics trajectories or a list
of 2D numpy arrays, each of shape ``(length_of_trajecotry, n_features)``.


.. autosummary::
    :toctree: _cluster/
    :nosignatures:

    KCenters
    KMeans
    KMedoids
    MiniBatchKMedoids
    RegularSpatial
    LandmarkAgglomerative
    AffinityPropagation
    GMM
    MeanShift
    MiniBatchKMeans
    SpectralClustering
    Ward


.. todo: Example of clustering

References
----------

.. [#f1] The "data", for MD, refers to snapshots of the structure of a molecular system at a given time point -- i.e the set of cartesian coordinates for all the atoms, or some mathematical transformation thereof.
.. [#f2] Aloise, Daniel, et al. `NP-hardness of Euclidean sum-of-squares clustering. <http://link.springer.com/article/10.1007/s10994-009-5103-0#page-1>`_ Machine Learning 75.2 (2009): 245-248.
.. [#f3] http://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions

.. vim: tw=75
