Developer Notes / Tools
=======================

How to do a release
-------------------

- Update `docs/changelog.rst`. Use the github view that shows all the
  commits to master since the last release to write it.

- Update the version number in `setup.py`, change `ISRELEASED` to `True`

- Update the version number in `devtools/conda-recipe/meta.yaml`

- Commit to master, and tag the release on github.

- **For version 3.4 only.** Verify that `versions.json` is being generated
  correctly and uploaded. Remove this line from this file afterwards.

- To push the source to PyPI, use `python setup.py sdist
  --formats=gztar,zip upload`

- File a pull request against the
  [conda-recipes](https://github.com/omnia-md/conda-recipes) repository.
  Conda binaries should be automatically built (for linux, windows
  eventually).

- Build conda binaries for windows somehow.

- Make an announcement on github / email

- After tagging the release, make a NEW commit that changes `ISRELEASED`
  back to `False` in `setup.py`

It's important that the version which is tagged on github for the release be
the one with the ISRELEASED flag in setup.py set to true.
