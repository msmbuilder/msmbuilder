.. _installation:

Installation
============

Dependencies
------------

Mixtape is a python package that heavily leans on other components of the scientific python ecosystem. To use mixtape, the following libraries and software will need to be installed.

    Linux, Mac OS X or Windows operating system
        We develop mainly on 64-bit Linux and Mac machines. Windows is not
        well supported.

    `Python <http://python.org>`_ >= 2.6
        The development package (``python-dev`` or ``python-devel``
        on most Linux distributions) is recommended (see just below).

    `NumPy <http://numpy.scipy.org/>`_ >= 1.6.0
    
    `SciPy <http://scipy.org>`_ >= 0.11.0
    
    `scikit-learn <http://sklearn.org>`_ >= 0.14.0
        Many of the models in mixtape build off base classes in scikit-learn.
    
    `MDTraj <http://mdtraj.org>`_ >= 0.8.0
        MDTraj is a library for handing molecular dynamics trajectories.
    
    `Pandas <http://pandas.pydata.org>`_ >= 0.9.0
        Pandas is pretty cool

    `cython <http://cython.org>`_ >= 0.18.0
        This is needed to compile the package.

    `cvxopt <http://cvxopt.org/>`_
        Only one module in mixtape uses cvxopt. TODO: make cvxopt optional.

Recommended packages:

    `nose <http://somethingaboutorange.com/mrl/projects/nose/>`_
        Recommended, to run Theano's test-suite.

    `Git <http://git-scm.com>`_
        To download the source code

Basic Installation
------------------

.. code-block::
    
    git clone https://github.com/rmcgibbo/mixtape
    cd mixtape
    python setup.py install