Contributing
============

MSMBuilder is a collaborative project and we welcome all contributions.

Users
-----

If you find a bug or have a question, consider opening an `issue
<https://github.com/msmbuilder/msmbuilder/issues>`_. We're mainly grad
student researchers, not software developers or support staff, so have
patience if someone doesn't respond immediately. Don't be afraid to "bump"
a thread if it's been a couple days.

If you are an MSMBuilder user, we encourage you to "watch" the issue
tracker. The developers will often solicit feedback about design decisions
and features. Your input is important! You can also contribute by answering
other users' questions.


Developers
----------

To contribute a bug-fix or improvement, submit a pull request. Make sure
the changed code is `pep8 <https://www.python.org/dev/peps/pep-0008/>`_
compliant. Add a simple test case that would have failed but now passes
with your contribution. Make a note of the change in the :ref:`changelog<changelog>`
by editing ``docs/changelog.rst``. If you need help or clarification on any
of these points, open a pull request and we'll be more than happy to help.

To contribute a new feature, submit a pull request. The code should be
`pep8 <https://www.python.org/dev/peps/pep-0008/>`_ compliant. Include a
suite of tests to (1) verify your feature is working as intended and (2)
will not be broken in the future. Describe the feature in the
:ref:`changelog<changelog>` by editing ``docs/changelog.rst``. Provide literature
citations if applicable. Add an example IPython notebook that demonstrates
your new feature. It should run quickly and make a pretty plot.  If you
need help or clarification on any of these points, open a pull request and
we'll be more than happy to help.


Tests
~~~~~

We use unit testing to make sure things work properly to start with, and
don't get broken in the future.  Tests are organized into files roughly
corresponding to ``msmbuilder`` modules in ``msmbuilder/tests/test_*.py``.
Each test script contains a number of short, self-contained functions
beginning with the name ``test_``. You can add new tests to an existing
file, or create a new file if it's appropriate. The new tests will
automatically get discovered and run by our continuous integration (CI)
workers. Don't use relative imports in the tests. Don't use docstrings in
the tests. ``nose`` (our test runner) will use the docstrings as the name
of the test, and this is often not what we want. Include pseudo-docstrings
as ordinary comments at the top of each function and name the test function
descriptively

To run the tests, install msmbuilder (``python setup.py install`` or
``python setup.py develop``) and then run ``nosetests msmbuilder``
**outside of the source directory**. It won't import the right thing if you
run from the source directory.


Examples
~~~~~~~~

Contribute an example IPython notebook by putting it in the ``examples/``
directory. We'll probably format it and do the incantations to make it show
up in the published docs. Clear all outputs before committing the notebook
in git.

To include an example notebook in the docs (contributors other than
core-developers ignore this): Create ``docs/examples/Notebook-Name.rst``::

    Notebook Pretty Title
    =====================

    .. notebook:: Notebook-Name

And add ``Notebook-Name`` to the toctree in ``docs/examples/index.rst``.
The rendering code is in ``docs/sphinxext/notebook_sphinxext.py``.
Presumably, we could streamline this process so you only have to deposit a
notebook in ``examples/`` and it will automatically get added to the docs.

