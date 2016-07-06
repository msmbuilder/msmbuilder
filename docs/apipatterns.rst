.. _apipatterns:
.. currentmodule:: msmbuilder

API Patterns
============

Models in msmbuilder inherit from base classes in `scikit-learn
<http://scikit-learn.org/stable/>`_, and follow a similar API. Like
sklearn, each type of model is a python class. Models are "fit" to data,
and can then "transform" data into a different representation. Unlike
sklearn, the data here is a *list* (or :ref:`dataset<datasets>`) of time-series
arrays or trajectories.

Hyperparameters
---------------

Hyperparameters are passed in via
the ``__init__`` method and set as instance attributes.

.. code-block:: python

    from msmbuilder.decomposition import tICA
    tica = tICA(gamma=0.05)
    tica.fit(...)


Fit
---

The estimation of model parameters is done in ``fit()``. In msmbuilder, the
``fit()`` method always accepts a ``list`` or
:func:`~msmbuilder.dataset.dataset` of 2-dimensional arrays as input data,
where each array represents a single timeseries (trajectory) and has a
shape of ``(length_of_trajectory, n_features)``. Some models can also
accept a list of MD trajectories (:class:`~md.Trajectory`) as opposed to a
list of arrays.


.. code-block:: python

    features = [np.load('traj-1-features.npy'), np.load('traj-2-featues.npy')]
    assert features[0].ndim == 2 and features[1].ndim == 2

    clusterer = KCenters(n_clusters=100)
    clusterer.fit(dataset)

.. note::

    This is different from sklearn. In sklearn, estimators take a **single**
    2D array as input in ``fit()``. Here we use a list of arrays or
    trajectories.  However, for many models, it's still quite easy to go
    between sklearn-style input and msmbuilder-style input, as shown in
    the following code block.


.. todo: move to example notebook?

.. code-block:: python

    import msmbuilder.cluster
    import sklearn.cluster

    X_sklearn = np.random.normal(size=(100, 10))  # sklearn style input: (n_samples, n_features)
    X_msmb = [X_sklearn]  # MSMBuilder style input: list of (n_samples, n_features)

    clusterer_sklearn = sklearn.cluster.KMeans(n_clusters=5)
    clusterer_sklearn.fit(X_sklearn)

    clusterer_msmb = msmbuilder.cluster.KMeans(n_clusters=5)
    clusterer_msmb.fit(X_msmb)


Some models like :class:`~tica.tICA` only require a single pass over the
data. In this case, use the ``partial_fit`` method, which can incrementally
learn the model one trajectory at a time and be more memory-efficient.

Attributes
----------

Parameters of the model which are **learned or estimated** during ``fit()``
are always set as instance attributes that are named with a trailing
underscore. This is merely a convention, and not a special Python syntax.

.. code-block:: python

    tica = tICA(gamma=0.05)
    tica.fit(...)

    # timescales is an estimated quantity, so it ends in an underscore
    print(tica.timescales_)


Transform
---------

Many models also implement a ``transform()`` method, which converts an
input dataset to an alternative representation. For example, the
``transform`` method of :ref:`featurizers<featurization>` takes as input a
list of trajectories and returns a list of 2D feature arrays.
:ref:`Clustering<cluster>` takes a list of 2D feature arrays and returns
a list of 1D sequences.

Pipelines
---------

The models in msmbuilder are designed to work together as part of a
:class:`sklearn.pipeline.Pipeline`

.. code-block:: python

    from msmbuilder.cluster import KMeans
    from msmbuilder.msm import MarkovStateModel
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('cluster', KMeans(n_clusters=100)),
        ('msm', MarkovStateModel())
    ])
    pipeline.fit(dataset)

.. vim: tw=75
