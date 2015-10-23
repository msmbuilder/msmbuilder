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
complete and up-to-date list of build, run, and test dependencies.

Frequently Asked Questions
--------------------------

**I get a wierd error during compilation with ``pip install`` or ``python
setup.py install``. What's wrong?**

We've received a couple `reports <https://github.com/msmbuilder/msmbuilder/issues/391>`_
of incorrect code generation on earlier versions of cython with python3.4. Try upgrading
to the latest verion of cython, and reinstalling. If that doesn't work, open an issue
on the github `issue tracker <https://github.com/msmbuilder/msmbuilder/issues>`_.

**Do I need Anaconda python? Can't I use the python that comes with my
operating like ``/usr/bin/python``?**

You can have multiple ``python`` installations on your computer which do
not interact with one another at all. The system python interpreter is used
by your operating system for some of its own programs, but is not the best
choice for data analysis or science.

We strongly recommend that you install Anaconda or Miniconda python
distribution and that you have the ``conda`` package manager available.

If you're interested in some of the details about packaging and scientific
python, see `this blog post by Travis Oliphant
<http://technicaldiscovery.blogspot.com/2013/12/why-i-promote-conda.html>`_.

.. vim: tw=75
