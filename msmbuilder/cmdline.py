"""Simple command line app framework for a single CLI script organized into
subcommands (like git <pull/clone/push/etc>) that each are basically separate
scripts.

Each subcommand is a class that inherits from Command, and specifies its parser
options (for argparse) as class attributes

Examples
--------
class Fetch(Command):
    description = 'Download objects and refs from another repository'

    # these two opts get translated to `parser.add_argument()` calls on
    # an `argparse.ArgumentParser` object for this subcommand
    opt1 = argument('-o', '--opt1', help='whatever')
    opt2 = argument('-f', '--opt2', help='something else')

    def __init__(self, args):
        # args is an argparse.Namespace, basically what gets returned
        # by ArgumentParser.parse_args() in a standard argparse application
        self.args = args

    def start(self):
        # the framework call this method as the entry point to the subcommand
        # after the user's invoked it.

if __name__ == '__main__':
    app = App(name='git', description='git command line client')
    # App automatically finds all of the Command subclasses
    app.start()
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import

import abc
import argparse
import copy
import inspect
import itertools
import os
import re
import sys
import textwrap

from six import with_metaclass, PY2

from . import version

try:
    import numpydoc.docscrape
except ImportError:
    print('-' * 35, file=sys.stderr)
    print('              ERROR', file=sys.stderr)
    print('-' * 35, file=sys.stderr)
    print('The package `numpydoc` is required.\n', file=sys.stderr)
    print('Try:', file=sys.stderr)
    print('  $ conda install numpydoc', file=sys.stderr)
    print('or', file=sys.stderr)
    print('  $ pip install numpydoc', file=sys.stderr)
    print('-' * 35, file=sys.stderr)
    raise

# Shim py3's inspect.Parameter class
try:
    from inspect import Parameter
except ImportError:
    class Parameter:
        POSITIONAL_OR_KEYWORD = 'lalalapositionalorkeyword'
        empty = 'dodododoempty'

        def __init__(self, name, kind):
            self.name = name
            self.kind = kind
            self.default = self.empty

        def replace(self, default):
            self.default = default
            return self

# Work around a very odd bug in pytables, where it destroys arguments in
# sys.argv when imported
# https://github.com/PyTables/PyTables/issues/405
SAVED_SYSARGV = copy.copy(sys.argv)

__all__ = ['argument', 'argument_group', 'Command', 'App', 'FlagAction',
           'MultipleIntAction']

#-----------------------------------------------------------------------------
# argparse types
#-----------------------------------------------------------------------------


class MultipleIntAction(argparse.Action):
    """An argparse Action to be used as an alternative to `nargs='+', type=int`
    (instead, you use `nargs='+', action=MultipleIntAction`. This allows the user
    to specify either a space separated list of ints (ala the former solution)
    _or_ a comma separated list"""

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            values = ' '.join(values)
        try:
            parsed = [int(x) for x in re.findall('[^,;\s]+', values)]
            setattr(namespace, self.dest, parsed)
        except ValueError:
            raise argparse.ArgumentError(self, 'Invalid list of integers: "%s"' % values)


class FlagAction(argparse.Action):
    # From http://bugs.python.org/issue8538

    def __init__(self, option_strings, dest, default=None,
                 required=False, help=None, metavar=None,
                 positive_prefixes=['--'], negative_prefixes=['--no-']):
        self.positive_strings = set()
        self.negative_strings = set()
        for string in option_strings:
            assert re.match(r'--[A-z]+', string)
            suffix = string[2:]
            for positive_prefix in positive_prefixes:
                self.positive_strings.add(positive_prefix + suffix)
            for negative_prefix in negative_prefixes:
                self.negative_strings.add(negative_prefix + suffix)
        strings = list(self.positive_strings | self.negative_strings)
        super(FlagAction, self).__init__(option_strings=strings, dest=dest,
                                         nargs=0, const=None, default=default, type=bool, choices=None,
                                         required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.positive_strings:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)

#-----------------------------------------------------------------------------
# Argument Declaration Class Attributes
#-----------------------------------------------------------------------------


class argument(object):
    """Wrapper for parser.add_argument"""

    def __init__(self, *args, **kwargs):
        self.parent = None
        self.args = args
        self.kwargs = kwargs
        self.registered = False

    def register(self, root):
        root.add_argument(*self.args, **self.kwargs)


class argument_group(object):
    """Wrapper for parser.add_argument_group"""

    def __init__(self, *args, **kwargs):
        self.parent = None
        self.args = args
        self.kwargs = kwargs
        self.children = []
        self.option_strings = {}

    def add_argument(self, *args, **kwargs):
        arg = argument(*args, **kwargs)
        arg.parent = self
        for short in args:
            if isinstance(short, str) and short.startswith('-'):
                self.option_strings[short] = arg
        self.children.append(arg)

    def remove_argument(self, name):
        arg = self.option_strings[name]
        if arg in self.children:
            self.children.remove(arg)

    def replace_argument(self, *args, **kwargs):
        for short in args:
            if isinstance(short, str) and short.startswith('-'):
                self.remove_argument(short)
        self.add_argument(*args, **kwargs)

    def add_mutually_exclusive_group(self, *args, **kwargs):
        group = mutually_exclusive_group(*args, **kwargs)
        group.parent = self
        self.children.append(group)
        return group

    def register(self, root):
        group = root.add_argument_group(*self.args, **self.kwargs)
        for x in self.children:
            x.register(group)


class mutually_exclusive_group(object):
    """Wrapper parser.add_mutually_exclusive_group"""

    def __init__(self, *args, **kwargs):
        self.parent = None
        self.args = args
        self.kwargs = kwargs
        self.arguments = []

    def add_argument(self, *args, **kwargs):
        arg = argument(*args, **kwargs)
        arg.parent = self
        self.arguments.append(arg)

    def register(self, root):
        group = root.add_mutually_exclusive_group(*self.args, **self.kwargs)
        for x in self.arguments:
            x.register(group)


#-----------------------------------------------------------------------------
# Command Classes
#-----------------------------------------------------------------------------

class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


class Command(with_metaclass(abc.ABCMeta, object)):
    # Set _concrete to true for all final classes in the heirarchy
    _concrete = False

    def __init__(self, args):
        pass

    @classmethod
    def _get_name(cls):
        if hasattr(cls, 'name'):
            return cls.name
        return cls.__name__

    @abc.abstractproperty
    def description(self):
        pass

    def error(self, msg):
        print(msg, file=sys.stderr)
        exit(1)


class NumpydocClassCommand(Command):
    """Subclass of Command that automatically populates arguments from
    the __init__ signature of a klass. klass must be a class whose
    class docstring is in the numpy format, including a "Parameters" section
    which gives the docstrings for the class's __init__ method. This will
    be parsed to create the argparse code for this command.

    When the class gets chosen on the command line executed, it's `start()`
    method will get called.

    Example
    -------
    >>> class HMMCommand(NumpydocClassCommand)
    >>>    klass = HMM
    >>>
    >>>    def start(self):
    >>>        print self.instance

    Notes
    -----
    argparse lets each argument have a `type`, which converts the input
    string as passed for that argument on the command line to the object
    that gets loaded up. This is by default inferred from the docstring, but
    can also be overridden by defining a method named _<attribute>_type, where
    <attribute> is the name of the argument (i.e. one of the arguments to
    klass.__init__)
    """

    # subclasses MUST override this
    klass = None

    # subclasses MAY override this
    example = None

    def __init__(self, args):
        # create the instance of `klass`

        init_args = [arg for arg in get_init_argspec(self.klass).parameters]
        kwargs = {}

        # these are all of the specified options from the command line
        # some of them correspond to __init__ args for self.klass, and
        # others are "extra" arguments that weren't part of klass

        # we do the "extra" arguments first, so we can use them when
        # loading init args

        for k, v in args.__dict__.items():
            if k not in init_args:
                # set the others as attributes on self
                setattr(self, k, v)

        for k, v in args.__dict__.items():
            if k in init_args:
                # put the ones for klass.__init__ in a dict
                kwargs[k] = v

                if hasattr(self, '_%s_type' % k):
                    kwargs[k] = getattr(self, '_%s_type' % k)(v)


        # make an instantiation of `klass`, populated with the requested
        # arguments from the command line
        self.instance = self.klass(**kwargs)

    @classmethod
    def _get_name(cls):
        # the default name of the subcommand will just be the name of the class
        return cls.klass.__name__

    @classmethod
    def _register_arguments(cls, subparser):
        """this is a special method that gets called to construct the argparse
        parser. it uses the python inspect module to introspect the __init__ method
        of `klass`, and add an argument for each parameter. it also uses numpydoc
        to read the class docstring of klass (which is supposed to be in numpydoc
        format) to get the help-text and type for each argument, as well as a
        description of the class."""

        assert cls.klass is not None

        sig = get_init_argspec(cls.klass)

        doc = numpydoc.docscrape.ClassDoc(cls.klass)
        # mapping from the name of the argument to the helptext
        helptext = {d[0]: ' '.join(d[2]) for d in doc['Parameters']}

        # mapping from the name of the argument to the type
        typemap = {d[0]: d[1].replace(',', ' ').split() for d in doc['Parameters']}

        # put all of these arguments into an argument group, to separate them
        # from other arguments on the subcommand
        group = argument_group('Parameters')

        for i, arg in enumerate(sig.parameters):
            if i == 0 and arg == 'self':
                continue

            # get default value
            kwargs = {}
            if sig.parameters[arg].default != Parameter.empty:
                kwargs['default'] = sig.parameters[arg].default
            else:
                kwargs['required'] = True

            if arg in helptext:
                # try to get some helptext
                kwargs['help'] = helptext[arg]

            # obviously this isn't an exaustive list, but try to make
            # reasonable argparse decisions based on the docstring.
            if arg in typemap:
                if 'list' in typemap[arg]:
                    kwargs['nargs'] = '+'
                if 'bool' in typemap[arg]:
                    kwargs['action'] = FlagAction

                if hasattr(cls, '_{}_type'.format(arg)):
                    # If the docstring *contains* the word float or int,
                    # parsing will fail for things not of that type
                    # even if a custom loader will eventually be used.
                    # Let's check for custom loaders here and set the type
                    # to str.
                    kwargs['type'] = str
                else:
                    basic_types = {'str': str, 'float': float, 'int': int}
                    for basic_type in basic_types:
                        if basic_type in typemap[arg]:
                            kwargs['type'] = basic_types[basic_type]
                            break

            group.add_argument('--{}'.format(arg), **kwargs)

        group.register(subparser)

    @classmethod
    def description(cls):
        doc = numpydoc.docscrape.ClassDoc(cls.klass)
        lines = []
        summary = ' '.join(doc['Summary'])
        if not summary.endswith('.'):
            summary += '.'

        lines.extend((summary, ''))
        lines.extend(doc['Extended Summary'])

        if len(doc['Notes']) > 0:
            lines.extend(('', 'Notes', '------'))
            lines.extend(doc['Notes'])
        if len(doc['References']) > 0:
            lines.extend(('', 'References', '----------'))
            lines.extend(doc['References'])
        if len(doc['See Also']) > 0:
            lines.extend(('', 'See Also', '--------'))
            for other in doc['See Also']:
                name = other[0]
                lines.append('%s:' % name)
                for l in textwrap.wrap(' '.join(other[1])):
                    lines.append('    %s' % l)

        if cls.example is not None:
            lines.extend(('', 'Example Command', '---------------'))
            for l in cls.example.splitlines():
                if l.startswith('    '):
                    lines.append(l[4:])

        return '\n'.join(lines)


class App(object):
    subcommand_dest = 'subcommand'

    def __init__(self, name, description, argv=None):
        self.name = name
        self.description = description
        if argv is None:
            argv = SAVED_SYSARGV[1:]
        if len(argv) == 0:
            argv.append('-h')
        self.parser = self._build_parser()

        # give a special "did you mean?" message if an invalid subcommand is
        # invoked
        cmdnames = []
        for act in self.parser._subparsers._actions:
            if hasattr(act, '_choices_actions'):
                cmdnames.extend((e.dest for e in act._choices_actions))
            else:
                cmdnames.extend(act.option_strings)

        if not argv[0] in cmdnames:
            import difflib
            lower2native = {s.lower(): s for s in cmdnames}
            didyoumean = difflib.get_close_matches(
                argv[0].lower(), lower2native.keys(), n=1, cutoff=0)[0]
            self.parser.error("invalid choice: '%s'. did you mean '%s'?" % (
                argv[0], lower2native[didyoumean]))
            sys.exit(1)

        self.args = self.parser.parse_args(argv)

    def start(self):
        name = getattr(self.args, self.subcommand_dest)
        exclude = [self.subcommand_dest]

        # figure out which command the user invoked
        klass = [k for k in self._subcommands() if k._get_name() == name][0]
        dct = ((k, v) for k, v in self.args.__dict__.items() if k not in exclude)
        args = argparse.Namespace(**dict(dct))
        instance = klass(args)
        # and then call start() on it
        instance.start()
        return instance

    def _build_parser(self):
        # Using a custom "MyHelpFormatter" to monkey-patch argparse into making
        # all of the subcommands get rendered in the help text on one line. To
        # do this, you need to increase the "action_max_length" argument which
        # puts more whitespace between the end of the action name and the start
        # of the helptext.
        fmt_class = lambda prog: MyHelpFormatter(
            prog, indent_increment=1, width=120, action_max_length=22)
        parser = argparse.ArgumentParser(
            description=self.description, formatter_class=fmt_class)
        parser.add_argument('-v', '--version',
            help="show program's version number and exit", action = 'version',
            version='msmbuilder %s' % version.full_version)

        subparsers = parser.add_subparsers(dest=self.subcommand_dest, title="commands", metavar="")

        def _key(klass):
            #return getattr(klass, '_group', klass)
            return getattr(klass, '_group')

        for title, group in itertools.groupby(sorted(self._subcommands(), key=_key), key=_key):
            # subparser = subparsers.add_argument_group(name)
            for klass in group:
                # http://stackoverflow.com/a/17124446/1079728
                klass_description = klass.description
                if callable(klass_description):
                    klass_description = klass_description()

                first_sentence = ' '.join(
                    ' '.join(re.split(r'(?<=[.;])\s', klass_description)[:1]).split())
                description = klass_description
                subparser = subparsers.add_parser(
                    klass._get_name(), help=first_sentence, description=description,
                    formatter_class=MyHelpFormatter)

                for v in (getattr(klass, e) for e in dir(klass)):
                    if isinstance(v, (argument, argument_group, mutually_exclusive_group)):
                        if v.parent is None:
                            v.register(subparser)
                if issubclass(klass, NumpydocClassCommand):
                    klass._register_arguments(subparser)

        return parser

    @classmethod
    def _subcommands(cls):
        for subclass in all_subclasses(Command):
            if subclass._concrete:
                yield subclass


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class MyHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):

    def __init__(self, *args, **kwargs):
        # to see what's going on here, you really have to look in the argparse source.
        # e.g. line 487 in argparse.py, where _action_max_length is used to set the
        # lateral position of the help text. This is really hacky.
        action_max_length = kwargs.pop('action_max_length', 0)
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._action_max_length = action_max_length

    def _get_help_string(self, action):
        help = action.help
        if action.default:
            return super(MyHelpFormatter, self)._get_help_string(action)
        return help


def rangetype(s):
    split = s.split(':')

    if len(split) == 2:
        return list(range(int(split[0]), int(split[1]) + 1))
    elif len(split) == 3:
        return list(range(int(split[0]), int(split[1]) + 1, int(split[2])))
    raise ValueError(s)


def exttype(suffix):
    """Type for use with argument(... type=) that will force a specific suffix
    Especially for output files, so that we can enforce the use of appropriate
    file-type specific suffixes"""
    def inner(s):
        if s == '':
            return s
        first, last = os.path.splitext(s)
        return first + suffix
    return inner


def stripquotestype(s):
    return s.strip('\'"')


def _shim_argspec(argspec):
    from collections import OrderedDict

    class Signature:
        def __init__(self):
            self.parameters = OrderedDict()

    sig = Signature()
    args, _, _, defaults = argspec
    for arg in args:
        sig.parameters[arg] = Parameter(arg, Parameter.POSITIONAL_OR_KEYWORD)

    if defaults is not None:
        last_args = list(sig.parameters.keys())[-len(defaults):]
        for arg, default in zip(last_args, defaults):
            sig.parameters[arg] = sig.parameters[arg].replace(default=default)

    return sig


def get_init_argspec(klass):
    """Wrapper around inspect.getargspec(klass.__init__) which, for cython
    classes uses an auxiliary '_init_argspec' method, since they don't play
    nice with the inspect module.

    By convention, a cython class should define the classmethod _init_argspec
    that, when called, returns what ``inspect.getargspec`` would be expected
    to return when called on that class's __init__ method.
    """
    if hasattr(klass, '_init_argspec'):
        return _shim_argspec(klass._init_argspec())
    elif PY2:
        return _shim_argspec(inspect.getargspec(klass.__init__))
    else:
        return inspect.signature(klass.__init__)
