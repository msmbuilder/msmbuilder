Changelog
=========

v3.2 (April 14, 2015)
---------------------

- `tICA` ignores too-short trajectories durring fitting instead of raising
  an exception
- New methods for sampling from MSM models
- Datasets can be opened in "append" mode
- Compatibility with sklearn 0.16
- `utils.dump` saves using the pickle protocol. `utils.load` is backwards
  compatible.
- The command line flag for featurizers `--out` is deprecated. Use
  `--transformed` instead. This is consistent with other command-line
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
