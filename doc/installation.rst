.. _installation:

Installation
============

Installation
------------

The preferred installation mechanism for ``msmbuilder`` is with ``conda``.

.. code-block:: bash

    $ conda install -c https://conda.binstar.org/omnia msmbuilder


If you don't have conda, or are new to scientific python, we recommend that
you download the `Anaconda scientific python
distribution <https://store.continuum.io/cshop/anaconda/>`_.


Dependencies
------------

.. I copied a lot of this formatting and text from the Theano docs
.. (http://deeplearning.net/software/theano/_sources/install.txt)
.. Thanks guys!

MSMBuilder is a python package that heavily leans on other components of the
scientific python ecosystem. To use msmbuilder, the following libraries and
software will need to be installed.

    Linux, Mac OS X, or Windows
        We develop mainly on 64-bit Linux and Mac machines. Windows _is_ supported,
        but may be a little rough around the edges.

    `Python <http://python.org>`_ >= 2.6
        The development package (``python-dev`` or ``python-devel``
        on most Linux distributions) is recommended.

    `NumPy <http://numpy.scipy.org/>`_ >= 1.6.0
        Numpy is the base package for numerical computing in python.

    `SciPy <http://scipy.org>`_ >= 0.11.0
        We use scipy for sparse matrix, numerical linear algebra and
        optimization.

    `scikit-learn <http://sklearn.org>`_ >= 0.14.0
        Many of the models in msmbuilder build off base classes in scikit-learn.

    `MDTraj <http://mdtraj.org>`_ >= 0.8.0
        MDTraj is a library for handing molecular dynamics trajectories.

    `Pandas <http://pandas.pydata.org>`_ >= 0.9.0
        Pandas is pretty cool

    `cython <http://cython.org>`_ >= 0.18.0
        This is needed to compile the package.

    `cvxopt <http://cvxopt.org/>`_
        Only one module in MSMBuilder uses cvxopt. TODO: make cvxopt optional.

Optional packages:

    `nose <http://somethingaboutorange.com/mrl/projects/nose/>`_
        Recommended, to run the test-suite.

    `Git <http://git-scm.com>`_
        To download the source code


Frequently Asked Questions
==========================

I get a wierd error during compilation with ``pip install`` or ``python setup.py install``. What's wrong?
---------------------------------------------------------------------------------------------------------

We've received a couple `reports <https://github.com/msmbuilder/msmbuilder/issues/391>`_
of incorrect code generation on earlier versions of cython with python3.4. Try upgrading
to the latest verion of cython, and reinstalling. If that doesn't work, open an issue
on the github `issue tracker <https://github.com/msmbuilder/msmbuilder/issues>`_.

Do I need Anaconda python? Can't I use the python that comes with my operating like ``/usr/bin/python``?
---------------------------------------------------------------------------------------------------------

You can have multiple ``python`` installations on your computer which do not
interact with one another at all. The system python interpreter is used by
your operating system for some of its own programs, but is not the best choice
for data analysis or science.

We strongly recommend that you install Anaconda or Miniconda python distribution
and that you have the ``conda`` package manager available.

If you're interested in some of the details about packaging and scientific
python, see `this blog post by Travis Oliphant
<http://technicaldiscovery.blogspot.com/2013/12/why-i-promote-conda.html>`_.
