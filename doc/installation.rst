.. _installation:

Installation
============

Basic Installation
------------------

.. code-block:: python

    git clone https://github.com/rmcgibbo/mixtape
    cd mixtape
    python setup.py install


Dependencies
------------

.. I copied a lot of this formatting and text from the Theano docs
.. (http://deeplearning.net/software/theano/_sources/install.txt)
.. Thanks guys!

Mixtape is a python package that heavily leans on other components of the
scientific python ecosystem. To use mixtape, the following libraries and
software will need to be installed.

    Linux, Mac OS X or Windows operating system
        We develop mainly on 64-bit Linux and Mac machines. Windows is not
        well supported.

    `Python <http://python.org>`_ >= 2.6
        The development package (``python-dev`` or ``python-devel``
        on most Linux distributions) is recommended.

    `NumPy <http://numpy.scipy.org/>`_ >= 1.6.0
        Numpy is the base package for numerical computing in python.

    `SciPy <http://scipy.org>`_ >= 0.11.0
        We use scipy for sparse matrix, numerical linear algebra and
        optimization.

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

Optional packages:

    `nose <http://somethingaboutorange.com/mrl/projects/nose/>`_
        Recommended, to run Theano's test-suite.

    `Git <http://git-scm.com>`_
        To download the source code

    `NVIDIA CUDA drivers and SDK`_
        Required for GPU code generation/execution. Only NVIDIA GPUs using
        32-bit floating point numbers are currently supported.


.. _NVIDIA CUDA drivers and SDK: http://developer.nvidia.com/object/gpucomputing.html
