.. _plugins:

Writing Plugins
===============

You can easily extend MSMBuilder by subclassing ``BaseEstimator`` or any of
its children. You can even build your plugin to work with the ``msmb``
command-line interface. 

1. Subclass ``cmdline.Command`` or any of its children. For example,
   if you want to expose a new Featurizer from the command line.

.. code-block:: python

    from msmbuilder.commands.featurizer import FeaturizerCommand
    class MyNiftyFeaturizerCommand(FeaturizerCommand):
        klass = MyNiftyFeaturizer
        _concrete = True

2. Provide your command as an "entry point" with ``setuptools``.
   Use ``"msmbuilder.commands"`` as the entry point.
   For example, in your ``setup.py``.

.. code-block:: python

    setup(
        ...
        entry_points={'msmbuilder.commands':
                           'niftyfeat = niftyfeat:MyNiftyFeaturizerCommand'
    )

See the 
`setuptools documentation <https://pythonhosted.org/setuptools/setuptools.html#extensible-applications-and-frameworks>`_
for more information.

.. vim: tw=75
