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

- `git checkout master && git fetch origin && git reset --hard origin/master`
- `git clean -fdx`
- Update the version number in `setup.py`, change `ISRELEASED` to `True`.
- Add the date of release to `docs/changelog.rst`, add a blurb.
- Commit and push to master. The commit should only include the version number changes and
  should be given a message like "Release 3.y.z".
- Tag that commit on GitHub. For version 3.y.z, the tag should be `3.y.z` (no "v" prefix)
  and the title should be "MSMBuilder 3.y" (no .z). You should copy-paste the changelog entry
  for the GitHub release notes. Beware of the transition from `rst` to markdown. In particular,
  you might have to change the headings from underlined to prefixed with `##`. You should
  also delete "hard-wrap" linebreaks because GitHub will keep them in! (and you don't want
  that). Use the preview tab.
- The docs will build. Make sure this is successful and they are live at msmbuilder.org.
  The docs will be sent to msmbuilder.org/3.y.z instead of development/ because you
  set `ISRELEASED`. You can cancel the Travis build triggered by the "tag" because docs
  are set to deploy only from `master`.
- Verify that [`versions.json`](http://msmbuilder.org/versions.json) was updated properly.
- Create the canonical source distribution using `python setup.py sdist --formats=gztar,zip`.
  Inspect the files in dist/ to make sure they look right.
- Upload to PyPI using `twine upload [path to sdist tar file] [path to sdist zip file]`.
  Make sure you upload both files in the same command. Note that removing files from PyPI
  means they can never be re-uploaded.
- File a pull request against the
  [conda-recipes](https://github.com/omnia-md/conda-recipes) repository.
  Use the PyPI link as the "source". Make sure the requirements match those
  in the msmbuilder recipe in `devtools/conda-recipe`. We don't want the package
  that gets tested with every pull request to differ from the one people actually get!
  Conda binaries should be automatically built with the `rc` tag.
- To test the release candidate, you can create a virtual environment like this:
  `conda create -n msmb-test -c omnia/label/rc msmbuilder`
  And then run:
  `source activate msmb-test`
  `nosetests -v msmbuilder --nologcapture`
- Once the tests have (successfully) completed, change the tag to main as follows:
  1. Go to [the conda page](https://anaconda.org/omnia/mdtraj/files)
  2. Filter by label, rc
  3. Use checkbox to select all
  4. Actions, move, main
- Make an announcement on the mailing list.

### Post-release

- Update the version number in `setup.py` to `3.(y+1).0.dev0`, change `ISRELEASED` to `False`.
- Add a new "development" entry in `docs/changelog.rst`.
- Commit and push to master.
- Make sure there is a 3.(y+1) milestone already created
- Create a new 3.(y+2) milestone [y is still the value of the release you just did]
- Close the 3.y milestone.
- Update this file (`devtools/README.md`) with anything you learned or
  changed during this release creation.
- Open an Issue for 3.(y+1) release schedule.

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

```
vim: tw=90
```
