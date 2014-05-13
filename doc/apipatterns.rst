.. _meta:
.. currentmodule:: mixtape

Mixtape API Patterns
====================

Mixtape API
-----------

Hyperparameters
~~~~~~~~~~~~~~
Models in mixtape inherit from base classes in `scikit-learn
<http://scikit-learn.org/stable/>`_, and follow a similar API. Like sklearn,
each type of model is a python class. Hyperparameters are passed in via
the ``__init__`` method and set as instance attributes.

.. code-block:: python

    from mixtape.tica import tICA
    tica = tICA(gamma=0.05)
    tica.fit(...)

    # change gamma and refit. the old state will be discarded
    tica.gamma = 0.01
    tica.fit(...)

Fit Signature
~~~~~~~~~~~~~
The heavy lifting to actually fit the model is done in ``fit()``. In mixtape the
``fit()`` method always accepts a ``list`` of 2-dimensional arrays as input data,
where each array represents a single timeseries / trajectory and has a shape of
``(length_of_trajectory, n_features)``. Some models can also accept a list of
MD trajectories (:class:`~md.Trajectory`) as opposed to a list of arrays.


.. code-block:: python

    dataset = [np.load('traj-1-features.npy'), np.load('traj-2-featues.npy')]
    assert dataset[0].ndim == 2 and dataset[1].ndim == 2

    clusterer = KCenters(n_clusters=100)
    clusterer.fit(dataset)

.. note::

    This is different from sklearn. In sklearn, estimators take a **single**
    2D array as input in ``fit()``. Here we use a list of arrays or
    trajectories.


Some models like :class:`~tica.tICA` which only require a single pass over the
data can also be fit in a potentially more memory efficient way, using the
``partial_fit`` method, which can incrementally learn the model one trajectory
at a time.

Learned Attributes
~~~~~~~~~~~~~~~~~~
Parameters of the model which are **learned or estimated** during ``fit()``
are always set as instance attributes that are named with a trailing underscore

.. code-block:: python

    tica = tICA(gamma=0.05)
    tica.fit(...)

    # timescales is an estimated quantity, so it ends in an underscore
    print(tica.timescales_)


Transformers
~~~~~~~~~~~~
Many models also implement a transform() method, which apply a transformation
to a dataset. [TODO: WRITE ME]

Pipelines
---------

The models in mixtape are designed to work together as part of a
:class:`sklearn.pipeline.Pipeline`

.. code-block:: python

    from mixtape.cluster import KMeans
    from mixtape.markovstatemodel import MarkovStateModel
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('cluster', KMeans(n_clusters=100)),
        ('msm', MarkovStateModel())
    ])
    pipeline.fit(dataset)

Serialization
-------------
1. pickle
2. joblib
3. [TODO: WRITE ME]
