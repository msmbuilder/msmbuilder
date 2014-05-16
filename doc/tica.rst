.. _tica:
.. currentmodule:: mixtape

Time-structure Independent Components Analysis (tICA)
=====================================================

Background
----------

.. figure:: _static/tica_vs_pca.png
   :height: 300 px
   :width: 300 px
   :align: right

   tICA compared to PCA (courtesy of C. R. Schwantes)


Time-structure independent components analysis (tICA) is a method for finding the slowest-relaxing degrees of freedom
in a time series data set which can be formed from linear combinations from a set of input degrees of freedom.

tICA can be used as a dimensionality reduction method, and in that capacity it is somewhat similar to PCA.
However whereas PCA finds high-variance linear combinations of the input degrees of freedom, tICA finds
high-autocorrelation linear combinations of the input degrees of freedom.


Algorithms
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ~tica.tICA
    ~sparsetica.SparseTICA

Combination with MSM
--------------------

While the tICs are themselves approximations to the dominant eigenfunctions
of the propagator / transfer operator, the approach taken in [#f1]_ and [#f2]_
is to "stack" tICA with :ref:`msm`. For example, in [#f2]_, Perez-Hernandez et
al. first measured the 66 atom-atom distances between a set of atoms in each
frame of their MD trajectories, and then use tICA to find the slowest 1, 4, and
10 linear combinations of these degrees of freedom and transform the
66-dimensional dataset into a 1, 4, or 10-dimensional dataset. Then, they apply
:class:`~mixtape.cluster.KMeans` to the resulting data and build an MSM.

Example
-------


.. code-block:: python

    from mixtape.tica import tICA
    from mixtape.cluster import KMeans
    from mixtape.markovstatemodel import MarkovStateModel
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('tica', tICA(n_components=4)),
        ('kmeans', KMeans(n_clusters=1000)),
        ('msm': MarkovStateModel(),
    ])

    # load a list of 2D arrays, each of shape (length_of_trajectory, n_features)
    dataset = ...

    pipeline.fit(dataset)


References
----------
.. [#f1] Schwantes, Christian R., and Vijay S. Pande. `Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9 <http://dx.doi.org/10.1021/ct300878a>`_ J. Chem Theory Comput. 9.4 (2013): 2000-2009.
.. [#f2] Perez-Hernandez, Guillermo, et al. `Identification of slow molecular order parameters for Markov model construction <http://dx.doi.org/10.1063/1.4811489>`_ J Chem. Phys (2013): 015102.
.. [#f3] Naritomi, Yusuke, and Sotaro Fuchigami. `Slow dynamics in protein fluctuations revealed by time-structure based independent component analysis: The case of domain motions <http://dx.doi.org/10.1063/1.3554380>`_ J. Chem. Phys. 134.6 (2011): 065101.
