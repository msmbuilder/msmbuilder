.. _io:
.. currentmodule:: msmbuilder.io

I/O
===

A new, comprehensive way of doing data input and output has been introduced
in MSMBuidler 3.6. The previous :ref:`dataset<datasets>` method will still be supported
and may be appropriate in certain cases, especially if your data can't all
fit in memory.

Dictionary of trajectories
--------------------------

MSMBuilder learns from and transforms a collection of sequences. While the time-ordering
of each sequence is important, the order of the sequences themselves has no special
meaning. For I/O, we treat collections of sequences as a dictionary mapping between arbitrary
keys and the sequences (which are probably 2D numpy arrays). Because our sequences
are time-series, we call them "trajectories", although they may not be in normal
Cartesian space.

The ``io`` module
assumes a python dictionary. Ideally, we would use our dictionary keys
as filenames for individual sequences saved as individual ``.npy`` files
on disk. In practice, python dictionaries can be any python object and filenames
must be unique strings. The ``io`` module have a mapping from python-object keys
to  filenames. It's important to note that
the converse (going from filename to python object) does not need to be codified.
That's because (in contrast to the :ref:`dataset<datasets>` approach) we persist the set
of keys in a separate metadata file. This file is saved using python's ``pickle``
protocol and can contain arbitrary python objects [1]_.

.. [1] We don't just serialize the whole python dictionary of sequences using
       ``pickle``, because it chokes on big numpy arrays.

Mapping keys to filenames
-------------------------

By default, ``msmbuilder.io`` can handle mapping of "normal" dictionary keys
to filenames. This should work well with strings, integers, or tuples of the above.

.. autofunction:: msmbuilder.io.io.default_key_to_path

Metadata
--------

Per-sequence information should be persisted in a ``pandas`` ``DataFrame``.
They ``index`` of the dataframe should be they keys used in the trajectory.
This dataframe is required for the saving and loading functions, as it serves
as the canonical list of keys.

Estimators
----------

Estimators are persisted using the generic ``pickle`` protocol.


Saving and loading
------------------

.. autosummary::
    :toctree: _io/

    load_trajs
    save_trajs
    load_meta
    save_meta
    backup
    render_meta
    load_generic
    save_generic
    itertrajs
    preload_tops
    preload_top


Gathering Metadata
------------------

Gathering trajectory metadata should come at the start of an MSM project
after you have collected and pre-processed your molecular dynamics trajectories.
We provide utilities for parsing metadata for common ways of organizing a set
of molecular dynamics trajectories


.. autosummary::
    :toctree: _io/

    gather_metadata
    GenericParser
    NumberedRunsParser
    HierarchyParser


Project Templates
-----------------

The ``msmb TempalteProject`` command-line command generates a set of example
scripts to serve as a framework for an MSM project. You can use this
functionality programatically.

.. autosummary::
    :toctree: _io/

    TemplateProject


The templates are stored in ``msmbuilder/project_templates``. They are jinja2
templates.

 - Python files can optionally be converted into IPython notebooks during template rendering.
   Indicate where cell breaks should happen with ``## Description goes here``
 - The hierarchy of the template project is *not* read from the ``msmbuilder/project_templates``
   source filesystem hierarchy. It's explicitly listed as a Python expression in ``msmbuilder.io``.
   If you add a new template file, make sure you list it in ``msmbuilder.io`` or it will not be rendered.
 - Templates can contain yaml "front matter". For some reason, jinja2 doesn't support this, so it is
   parsed explicitly by MSMBuilder. Include the yaml as the last element in the file's docstring under
   a numpydoc heading "Meta"::
    
    Meta
    ----
    depends:
      - meta.pandas.pickl
    arbitrary_key:
      - arbitrary data

 - MSMBuilder defines some variables for use in your templates including ``{{header}}`` and ``{{date}}``.
   For a complete list, check the rendering code.
 - Plotting scripts should include the following macros **before** any imports::

    # ? include "plot_header.template"
    # ? from "plot_macros.template" import xdg_open with context

   This will set up matplotlib to use the correct backend. Add::
    
    # {{xdg_open('filename.pdf')}}

   to have a call to xdg-open inserted based on user configuration.

    

