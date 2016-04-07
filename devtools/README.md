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
- Fix and close all open issues and PRs with the release's milestone.

### Release

- `git checkout master && git reset --hard origin/master`
- `git clean -fdx`
- Update the version number in `setup.py`, change `ISRELEASED` to `True`.
- Update the version number in `devtools/conda-recipe/meta.yaml`.
- Add the date of release to `docs/changelog.rst`.
- Commit and push to master. The commit should
  only include the version number changes and should be given a message like
  "Release 3.y.z".
- Tag that commit on GitHub. For version 3.y.z, the tag should be `3.y.z` (no "v" prefix)
  and the title should be "MSMBuilder 3.y" (no .z). You should copy-paste the changelog entry
  for the GitHub release notes. Beware of the transition from `rst` to markdown. In particular,
  you might have to change the headings from underlined to prefixed with `##`.
- The docs will build. Make sure this is successful and they are live at msmbuilder.org.
  The docs will be sent to msmbuilder.org/3.y.z instead of development/ because you
  set `ISRELEASED`.
- Verify that `versions.json` was updated properly.
- Create the canonical source distribution using `python setup.py sdist --formats=gztar,zip`.
  Inspect the files in dist/ to make sure they look right.
- Upload to PyPI using `twine upload [path to sdist files]`.
- File a pull request against the
  [conda-recipes](https://github.com/omnia-md/conda-recipes) repository.
  Use the PyPI link as the "source". Make sure the requirements match those
  in the msmbuilder-dev recipe in `devtools/conda-recipe`. We don't want the package
  that gets tested with every pull request to differ from the one people actually get!
  Conda binaries should be automatically built.
- Make an announcement on the mailing list.

### Post-release

- Update the version number in `setup.py`, change `ISRELEASED` to `False`.
- Update the version number in `devtools/conda-recipe/meta.yaml`.
- Add a new "development" entry in `docs/changelog.rst`.
- Commit and push to master.
- Make sure there is a 3.(y+1) milestone already created
- Create a new 3.(y+2) milestone [y is still the value of the release you just did]
- Close the 3.y milestone.
- Update this file (`devtools/README.md`) with anything you learned or
  changed during this release creation.

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
