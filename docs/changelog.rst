.. _changelog:

Changelog
=========

v3.9 (development)
------------------

API Changes
~~~~~~~~~~~

New Features
~~~~~~~~~~~~
- Added new featurizer ```FeatureSlicer```. ```FeatureSlicer``` can slice the ouput of
regular featurizer objects to just the required indices.
- Generalized ```KappaAngleFeaturizer``` to be able compute the angles between arbitrarily
offset CA atoms.
- Added functions to compute error bars for transition probabilities to account for
  finite sampling, and sample transition matrices from these error distributions (i.e.
  bootstrapping). Located in ```msmbuilder.msm.validation.transmat_errorbar```.
- Added new featurizer ```FeatureSlicer```. ```FeatureSlicer``` can slice
  the ouput of regular featurizer objects to just the required indices
  (gh-1022).
- Added functions to compute error bars for transition probabilities to account
  for finite sampling, and sample transition matrices from these error 
  distributions (i.e. bootstrapping). Located in
  ```msmbuilder.msm.validation.transmat_errorbar``` (gh-1010).
- Added methods for computing the Kullbeck-Leibler, symmetric KL, and 
  Jensen-Shannon divergences of probability distributions, arrays thereof,
  or flattened MSM objects. The array and (flattened) MSM metrics are
  compatible with the custom distance function in ```LandmarkAgglomerative```
  (gh-1035).
- Added minimum variance cluster analysis (MVCA) for macrostating to
  msmbuilder.lumping (gh-1045).
  
Improvements
~~~~~~~~~~~~
- ```FeatureSelector``` is now compatible with Tree-structure Parzen Estimator
  method in Osprey (gh-1018).
- Fixed bug in ```from_msm``` method for ```PCCA``` and ```PCCAPlus``` which
  now allows a ```PCCAPlus``` objective function to be specified (gh-1036).
- ```msmbuilder.io.sampling.sample_dimension``` with ```scheme='edge'``` now works properly. (#1043)
- Changed zippy_maker code so that ```Featurizer.describe_features``` will
  return ordered unique lists to make reading and subselecting features easier.


v3.8 (April 26, 2017)
---------------------

We're pleased to annoounce the release of MSMBuilder 3.8. This release
features updates and improvements to contact featurizers, kernel tICA, HMMs,
and preprocessing. There are also some bugfixes and API hygiene improements.
We recommend all users upgrade to MSMBuilder 3.8.

API Changes
~~~~~~~~~~~

New Features
~~~~~~~~~~~~

- ``ContactFeaturizer`` now lets you use a soft_min option for closest
contact distances.

Improvements
~~~~~~~~~~~~

- The ``stride`` parameter in ``KernelTICA`` now works as intended to
automatically generate a set of landmark points (gh-972).
- The ``contacts`` parameter in ``CommonContactFeaturizer`` now performs as the
contacts method in regular ``ContactFeaturizer`` albeit after validating all
the contacts.
- ``GaussianHMM`` and ``VonMisesHMM`` are now compatible with
``sklearn.pipeline.Pipeline`` workflows (gh-980).
- ``msmbuilder.preprocessing`` is now compatible with
``sklearn.pipeline.Pipeline`` workflows (gh-987).
- Fixed error in pickling HMMs (gh-996).


v3.7 (January 26, 2017)
-----------------------

We're pleased to announce the release of MSMBuilder 3.7. This release
introduces several new featurizers that can handle multiple sequences or
multiple chains within a topology file. There are also some bugfixes and
API hygiene improvements. We recommend all users upgrade to MSMBuilder 3.7.

API Changes
~~~~~~~~~~~

- ``TrajFeatureUnion`` and ``SubsetFeatureUnion`` have been removed due to
  incompatibilities with the ``scikit-learn`` API.

New Features
~~~~~~~~~~~~

- ``KSparseTICA`` lets you specify the number of non-zero entries, ``k``
  rather than a regularization strength (gh-916).
- ``BootStrapMarkovStateModel`` optionally saves all the models that it
  generates (gh-919).
- ``tICA`` supports commute mapping (see 10.1021/acs.jctc.6b00762)
  (gh-925).
- ``CommonContactFeaturizer`` featurizes different trajectories with
  different topologies using a common set of inter-residue contacts
  (gh-876).
- ``msmbuilder.tpt.mfpt.mfpts`` can now compute distributions of MFPTs, accounting
  for the model error due to finite sampling.
- Three new featurization schemes for protein-ligand trajectories are
  now available: ``LigandContactFeaturizer``,
  ``BinaryLigandContactFeaturizer``, and ``LigandRMSDFeaturizer`` (gh-883).

Improvements
~~~~~~~~~~~~

- Compatibility with scikit-learn 0.18 (gh-915).
- ``FeatureSelector`` feature order is deterministic (gh-920).
- ``SASAFeaturizer`` supports the ``describe_features`` method (gh-913).
- All ``LandmarkAgglomerative`` clusterers now have ``cluster_centers_`` except
  when ``metric = rmsd`` (gh-958)


v3.6 (September 15, 2016)
-------------------------

We're pleased to announce the release of MSMBuilder 3.6. This release
introduces project templating and a whole host of new ``sklearn`` estimators.
There are also some bugfixes and API hygiene improvements. We recommend all
users upgrade to MSMBuilder 3.6.

API Changes
~~~~~~~~~~~

- ``version.short_version`` is now 3.y instead of 3.y.z (gh-829).
- ``weighted_transform`` is no longer supported in tICA methods (gh-807). Please
  used ``kinetic_mapping``.
- The cached filenames and formats for DoubleWell, QuadWell,
  and MullerPotential example datasets have changed. The API through
  ``msmbuilder.example_datasets`` is still the same, but the data may
  be re-generated instead of using a cached version from a previous installation
  of MSMBuilder (gh-854).
- The alias for Ward clustering has been removed. Modelers should now use
  ``LandmarkAgglomerative(linkage='ward')`` (gh-874). Ward clustering is also
  available in ``AgglomerativeClustering``, but without a prediction algorithm.

New Features
~~~~~~~~~~~~

- ``Butterworth``, ``DoubleEWMA``, ``StandardScaler``, ``RobustScaler`` are
  available via the command line (gh-895).
- ``BinaryContactFeaturizer`` featurizes a trajectory into a
  boolean array corresponding to whether each residue-residue
  distance is below a cutoff (gh-798).
- ``LogisticContactFeaturizer`` produces a logistic transform
  of residue-residue distances about a center distance (#798).
- ``FactorAnalysis``, ``FastICA``, and ``KernelPCA`` are available in the
  ``decomposition`` module (gh-807).
- ``Butterworth``, ``EWMA``, and ``DoubleEWMA`` are available in the
  ``preprocessing`` module (gh-818).
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
  ``ward`` objective function has been developed and implemented (gh-874).

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
