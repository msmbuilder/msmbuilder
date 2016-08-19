.. _changelog:

Changelog
=========

v3.6 (Development)
------------------

This is the current development version of MSMBuilder

API Changes
~~~~~~~~~~~

- ``version.short_version`` is now 3.y instead of 3.y.z (#829).
- ``weighted_transform`` is no longer supported in tICA methods (#807). Please
  used ``kinetic_mapping``.
- The cached filenames and formats for DoubleWell, QuadWell,
  and MullerPotential example datasets have changed. The API through
  ``msmbuilder.example_datasets`` is still the same, but the data may
  be re-generated instead of using a cached version from a previous installation
  of MSMBuilder (gh-854).

New Features
~~~~~~~~~~~~

- ``BinaryContactFeaturizer`` featurizes a trajectory into a
  boolean array corresponding to whether each residue-residue
  distance is below a cutoff (#798).
- ``LogisticContactFeaturizer`` produces a logistic transform
  of residue-residue distances about a center distance (#798).
- ``FactorAnalysis``, ``FastICA``, and ``KernelPCA`` are available in the
  ``decomposition`` module (#807).
- ``Butterworth``, ``EWMA``, and ``DoubleEWMA`` are available in the
  ``preprocessing`` module (#818).
- We encourage users to download the ``msmb_data`` conda package to easily
  install example data. The data can be loaded through existing methods
  in ``msmbuilder.example_datasets`` (gh-854, gh-867).
- An example dataset ``MinimalFsPeptide`` is available. This is a strided
  version of the existing ``FsPeptide`` dataset. We use it for testing,
  when a fully-converged dataset is not required (gh-867).
- Project templates! Read the new tutorial or the :ref:`io` page for
  details (gh-768).
- ``LandmarkAgglomerative`` clustering now features the ``ward`` linkage
  option. An algorithm for predicting cluster assignments with the
  ``ward`` objective function has been developed and implemented (#874).

Improvements
~~~~~~~~~~~~

- Remove a unicode character from ``ktica.py`` (gh-833)
- ``msmbuilder.decomposition.KernelTICA`` now includes all parameters in its
  ``__init__``, making it compatible with Osprey (gh-823).
- ``msmbuilder.tpt`` methods can now handle ``BayesianMarkovStateModels`` as
  input. Please note that we still do not recommend using this module with
  ``BootStrapMarkovStateModel``.


v3.5 (June 14, 2016)
--------------------

We're pleased to announce the release of MSMBuilder 3.5. This release
wraps more relevant ``sklearn`` estimators and transformers. There are
also some bugfixes and API hygiene improvements. We recommend all users
upgrade to MSMBuilder 3.5.

API Changes
~~~~~~~~~~~

- ``msmbuilder.featurizer.FeatureUnion`` is now deprecated. Please use
  ``msmbuilder.feature_selection.FeatureSelector`` instead (#799).
- ``msmbuilder.feature_extraction`` has been added to conform to the
  ``scikit-learn`` API. This is essentially an alias of
  ``msmbuilder.featurizer`` (#799).

New Features
~~~~~~~~~~~~

 - ``KernelTICA``, ``Nystroem``, and ``LandmarkNystroem`` are available in the
   ``decomposition`` module (#807).

 - ``FeatureSelector`` and ``VarianceThreshold`` are available in the
   ``feature_selection`` module (#799).

 - ``SparsePCA`` and ``MiniBatchSparsePCA`` are available in the
   ``decomposition`` module (#791).

 - ``Binarizer``, ``FunctionTransformer``, ``Imputer``, ``KernelCenterer``,
   ``LabelBinarizer``, ``MultiLabelBinarizer``, ``MinMaxScaler``,
   ``MaxAbsScaler``, ``Normalizer``, ``RobustScaler``, ``StandardScaler``,
   and ``PolynomialFeatures`` are available in the ``preprocessing``
   module (#796).


Improvements
~~~~~~~~~~~~

- Fix a compilation error on gcc 5 (#783)
- Fix pickle-ing of ``ContinuousTimeMSM``. The ``optimizer_state_``
  parameter is not saved (#822).


v3.4 (March 29, 2016)
---------------------

We're pleased to announce MSMBuilder 3.4. It contains a plethora of new
features, bug fixes, and improvements.

API Changes
~~~~~~~~~~~

- Range-based slicing on dataset objects is no longer allowed. Keys in the
  dataset object don't have to be continuous. The empty slice, e.g. ``ds[:]``
  loads all trajectories in a list (#610).
- Ward clustering has been renamed AgglomerativeClustering in scikit-learn.
  Please use the new msmbuilder wrapper class AgglomerativeClustering. An
  alias for Ward has been made available (#685).
- ``PCCA.trimmed_microstates_to_macrostates`` has been removed. This
  dictionary was actually keyed by *untrimmed* microstate labels.
  ``PCCA.transform`` would throw an exception when operating on a system
  with trimming because it was using this misleading dictionary. Please use
  ``pcca.microstate_mapping_`` for this functionality (#709).
- ``UnionDataset`` has been removed after deprecation in 3.3. Please use
  ``FeatureUnion`` instead (#671).
- ``SubsetFeaturizer`` and ilk have been removed from the
  ``msmbuilder.featurizer`` namespace. Please import them from
  ``msmbuilder.featurizer.subset`` (#738).
- ``FirstSlicer`` has been removed. Use ``Slicer(first=x)`` for the same
  functionality (#738).
- ``msmbuilder.featurizer.load`` has been removed. ``Featurizer.save``
  has been removed. Please use ``utils.load``, ``utils.dump`` (#738).


New Features
~~~~~~~~~~~~

- Dataset objects can call, ``fit_transform_with()`` to simplify the
  common pattern of applying an estimator to a dataset object to produce a
  new dataset object (#610).
- ``kinetic_mapping`` is a new option to ``tICA``. It's similar to
  ``weighted_transform``, but based on a better theoretical framework.
  ``weighted_transform`` is deprecated (#766).
- ``VonMisesFeaturizer`` uses soft bins around the unit-circle to give an
  alternate representation of dihedral angles (#744).
- ``MarkovStateModel`` has a ``partial_transform()`` method (#707).
- ``KappaAngleFeaturizer`` is available via the command line (#681).
- ``MarkovStateModel`` has a new attribute, ``percent_retained_``, for
  ergodic trimming (#689).
- ``AlphaAngleFeaturizer`` computes the dihedral angles between alpha
  carbons (#691).
- ``FunctionFeaturizer`` computes features based on an arbitrary Python
  function or callable (#717).
- Automatic State Partitioning (APM) uses kinetic information to cluster
  conformations (#748).


Improvements
~~~~~~~~~~~~

- Consistent counts setup and ergodic cutoff across various flavors of
  Markov models (#718, #729, #701, #705).
- Tests no longer depend on ``sklearn.hmm``, which has been removed (#690).
- Improvements to ``RSMDFeaturizer`` (#695, #764).
- ``SparseTICA`` is completely re-written with large performance
  improvements when dealing with large numbers of features (#704).
- Links for downloading example data are un-broken after figshare
  changed URLs (#751).



v3.3 (August 27, 2015)
----------------------

We're pleased to announce the release of MSMBuilder v3.3.0. The focus of this
release is a completely re-written module for constructing HMMs as well as bug
fixes and incremental improvements.

API Changes
~~~~~~~~~~~

- ``FeatureUnion`` is an estimator that deprecates the functionality of
  ``UnionDataset``. Passing a list of paths to ``dataset()`` will no longer
  automatically yield a ``UnionDataset``. This behavior is still available by
  specifying ``fmt="dir-npy-union"``, but is deprecated (#611).
- The command line flag for featurizers ``--out`` (deprecated in 3.2) now saves
  the featurizer as a pickle file (#546). Please use ``--transformed`` for the
  old behavior. This is consistent with other command-line commands.
- The default number of timescales in ``MarkovStateModel`` is now one less than
  the number of states (was 10). This addresses some bugs with
  ``implied_timescales`` and PCCA(+) (#603).

New Features
~~~~~~~~~~~~

- ``GaussianHMM`` and ``VonMisesHMM`` is rewritten to feature higher code reuse
  and code quality (#583, #582, #584, #572, #570).
- ``KDTree`` can find n nearest points to e.g. a cluster center (#599).
- ``Slicer`` featurizer can slice feature arrays as part of a pipeline
  (#567).

Improvements
~~~~~~~~~~~~

- ``PCCAPlus`` is compatible with scipy 0.16 (#620).
- Documentation improvements (#618, #608, #604, #602)
- Test improvements, especially for Windows (#593, #590, #588, #579, #578,
  #577, #576)
- Bug fix: ``MarkovStateModel.sample()`` produced trajectories of incorrect
  length. This function is still deprecated (#556).
- Bug fix: The muller example dataset did not respect users' specifications for
  initial coordinates (#631).
- ``MarkovStateModel.draw_samples`` failed if discrete trajectories did not
  contain every possible state (#638). Function can now accept a single
  trajectory, as well as a list of them.
- ``SuperposeFeaturizer`` now respects the topology argument when loading the
  reference trajectory (#555).

v3.2 (April 14, 2015)
---------------------

- ``tICA`` ignores too-short trajectories during fitting instead of raising
  an exception
- New methods for sampling from MSM models
- Datasets can be opened in "append" mode
- Compatibility with scipy 0.16
- ``utils.dump`` saves using the pickle protocol. ``utils.load`` is backwards
  compatible.
- The command line flag for featurizers ``--out`` is deprecated. Use
  ``--transformed`` instead. This is consistent with other command-line
  commands.
- Bug fixes

v3.1 (Feb 27, 2015)
-------------------

- Numerous improvements to ``ContinuousTimeMSM`` optimization
- Switch ``ContinuousTimeMSM.score`` to transmat-style GMRQ
- New example dataset with Muller potential
- Assorted bug fixes in the command line layer

v3.0.1 (January 9, 2015)
------------------------

- Fix missing file on PyPI.


v3.0.0 (January 9, 2015)
------------------------

MSMBuilder 3.0 is a complete rewrite of our `previous work
<https://github.com/msmbuilder/msmbuilder-legacy>`_. The focus is on power
and extensibility, with a much wider class of estimators and models
supported throughout the codebase. All users are encouraged to switch to
MSMBuilder 3.0.  Pre-release versions of MSMBuilder 3.0 were called
mixtape.
