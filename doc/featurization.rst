.. _featurization:
.. currentmodule:: msmbuilder.featurizer


Featurization
=============

Many algorithms require that the input data be vectors in a (euclidean)
vector space. This includes :class:`~msmbuilder.cluster.KMeans` clustering,
:class:`~msmbuilder.decomposition.tICA`, and others.  In the absense of
vector data, other algorithms like :class:`~msmbuilder.cluster.KCenters`
clustering require that a pairwise distance.

One of the complexities of featurizing molecular dynamics trajectories is
that during a simulation, the system is generally permitted to tumble
(rotate) in 3D. The timescale for this tumbling is generally fast, and
there's usually no special rotational reference frame. It's often desirable
to remove rotational motion either via featurization or via a distance
metric that is insensitive to rotations.  This can be done by featurizing
with internal coordinates.

Featurizations
--------------

.. autosummary::
    :toctree: generated/

    AtomPairsFeaturizer
    ContactFeaturizer
    DRIDFeaturizer
    DihedralFeaturizer
    GaussianSolventFeaturizer
    RMSDFeaturizer
    RawPositionsFeaturizer
    SuperposeFeaturizer


Distance Metrics
----------------

Some :ref:`clustering <cluster>` methods let you pass in a custom distance
metric. In that case, the input to ``fit()`` may be a list of MD
trajectories instead of a list of numpy arrays. Clustering methods that
allow this currently include :class:`~msmbuilder.cluster.KCenters` and
:class:`~msmbuilder.cluster.LandmarkAgglomerative`. See their documentation
for details.


.. vim: tw=75
