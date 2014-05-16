.. _featurization:
.. currentmodule:: mixtape.featurizer


Featurization and Distance Metrics
==================================

Background
----------

Many analyses require that the input data be vectors in a (euclidean) vector
space. This includes :class:`~mixtape.cluster.KMeans` clustering,
:class:`~mixtape.tica.tICA` and others. Furthermore, other analyses like
:class:`~mixtape.cluster.KCenters` clustering require that, if the data are not
vectors, that a pairwise distance metric be supplied.

One of the complexities of featurizing molecular dynamics trajectories is that
during a simulation, the system is generally permitted to tumble (rotate)
in 3D, and the timescale for this tumbling is pretty fast. For a protein in bulk solvent, there's no special rotational reference frame either. So it's often desirable to remove rotational motion either via featurization or via
a distance metric that is insensitive to rotations. This can be done by featurizing with internal coordinates.

Featurizations
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

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

Some :ref:`clustering <cluster>` methods let you pass in a custom distance metric. In that case, the input to ``fit()`` may be a list of MD trajectories instead of a list of numpy arrays. Clustering methods that allow this currently include :class:`~mixtape.cluster.KCenters` and :class:`~mixtape.cluster.LandmarkAgglomerative`. See their documentation for details.
