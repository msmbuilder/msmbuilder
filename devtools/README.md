Developer Notes / Tools
=======================

How to do a release
-------------------

### Pre-release

- Tag issues and pull requests with a milestone for the particular version.
- Use the "changelog" tag to tag pull requests that need to be recorded in `docs/changelog.rst`.
  You've been encouraging people to update the changelog *with* their pull requests, though, right?
- Update `docs/changelog.rst`. Change the tag from "changelog" to "changelogged".
- Optionally create an issue about cutting the release.
- Fix and close all open issues and PRs with the release's milesteon

### Release

- `git pull origin master`
- `git clean -fdx`
- Update the version number in `setup.py`, change `ISRELEASED` to `True`
- Update the version number in `devtools/conda-recipe/meta.yaml`
- Commit and push to masterThe commit should
  only include the version number changes and should be given a message like
  "Release 3.x"
- Tag that commit on GitHub. For version 3.y.z, the tag should be `3.y.z` (no "v" prefix)
  and the title should be "MSMBuilder 3.y" (no .z)
- The docs will build. Make sure this is successful and they are live at msmbuilder.org.
  The docs will be sent to msmbuilder.org/3.y.z instead of development/ because you
  set `ISRELEASED`.
- Verify that `versions.json` was updated properly.
- To push the source to PyPI, use `python setup.py sdist --formats=gztar,zip upload`
  TODO: use twine
- File a pull request against the
  [conda-recipes](https://github.com/omnia-md/conda-recipes) repository.
  Conda binaries should be automatically built.
- Make an announcement on the mailing list

### Post-release

- Update the version number in `setup.py`, change `ISRELEASED` to `False`.
- Update the version number in `devtools/conda-recipe/meta.yaml`.
- Commit and push to master.

### Point releases

If you want to include a minor or important fix, you can create a point release.
For version 3.y.z, this would mean bumping `z`.

- `git checkout 3.y.0` (the tag)
- `git checkout -b 3.y` (make 3.y branch)
- Make a commit that updates the versions, isreleased in setup.py and conda recipe.
  This time, change to `3.y.(z+1).dev0` instead of `3.(y+1).0.dev0`
- `git push origin 3.y -u`
- Backport or cherry-pick the fixes you need.
- Go through the above for creating a release. Make sure you tag
  the commit on the 3.y branch. If you don't want release notes
  (e.g. for a really minor fix), you can create an unannotated tag.
