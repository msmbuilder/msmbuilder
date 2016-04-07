.. _featurization:
.. currentmodule:: msmbuilder.featurizer


Featurization
=============

Many algorithms require that the input data be vectors in a (euclidean)
vector space. This includes :class:`~msmbuilder.cluster.KMeans` clustering,
:class:`~msmbuilder.decomposition.tICA`, and others.

Since there's usually no special rotational or translational reference
frame in an MD simulation, it's often desirable to remove rotational and
translational motion via featurization that is insensitive to rotations and
translations. 

Featurizations
--------------

.. autosummary::
    :toctree: _featurization/

    AtomPairsFeaturizer
    ContactFeaturizer
    DRIDFeaturizer
    DihedralFeaturizer
    GaussianSolventFeaturizer
    RMSDFeaturizer
    RawPositionsFeaturizer
    SuperposeFeaturizer


Alternative to Featurization
----------------------------

Many algorithms require vectorizable data. Other algorithms only require a
pairwise distance metric, e.g. RMSD between two protein conformations. In
general, you can define a pairwise distance among vectorized data, but you
cannot embed data into a vector space only from pairwise distance.

Some :ref:`clustering <cluster>` methods let you use an arbitrary distance
metric, including RMSD. In this case, the input to ``fit()`` may be a list
of MD trajectories instead of a list of numpy arrays. Clustering methods
that allow this currently include :class:`~msmbuilder.cluster.KCenters` and
:class:`~msmbuilder.cluster.KMedoids`.

.. vim: tw=75
