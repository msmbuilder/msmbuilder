.. _installation:

Installation
============

The preferred installation mechanism for ``msmbuilder`` is with ``conda``.

.. code-block:: bash

    $ conda install -c omnia msmbuilder


If you don't have conda, or are new to scientific python, we recommend that
you download the `Anaconda scientific python distribution
<https://store.continuum.io/cshop/anaconda/>`_.


From Source
-----------

MSMBuilder is a python package that heavily leans on other components of the
scientific python ecosystem. See ``devtools/conda-recipe/meta.yaml`` for a
complete and up-to-date list of build, run, and test dependencies. When you
are sure the dependencies are satisfied you can install from PyPI

.. code-block:: bash

    $ pip install msmbuilder

or from source

.. code-block:: bash

    $ git clone git@github.com:msmbuilder/msmbuilder
    $ cd msmbuilder/
    $ pip install .
    $ # (or: python setup.py install)

Frequently Asked Questions
--------------------------

**Do I need Anaconda python? Can't I use the python that comes with my
operating like /usr/bin/python?**

You can have multiple ``python`` installations on your computer which do
not interact with one another at all. The system python interpreter is used
by your operating system for some of its own programs but is not the best
choice for data analysis or science.

We strongly recommend that you install Anaconda or Miniconda python
distribution and that you have the ``conda`` package manager available.

If you're interested in some of the details about packaging and scientific
python, see `this blog post by Travis Oliphant
<http://technicaldiscovery.blogspot.com/2013/12/why-i-promote-conda.html>`_.

.. vim: tw=75
