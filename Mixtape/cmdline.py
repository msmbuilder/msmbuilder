#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import sys
import abc
import argparse

__all__ = ['argument', 'argument_group', 'Command', 'App']

#-----------------------------------------------------------------------------
# Argument stuff
#-----------------------------------------------------------------------------


class argument(object):
    def __init__(self, *args, **kwargs):
        self.parent = None
        self.args = args
        self.kwargs = kwargs
        self.registered = False
        
    def register(self, root):
        root.add_argument(*self.args, **self.kwargs)


class argument_group(object):
    def __init__(self, *args, **kwargs):
        self.parent = None
        self.args = args
        self.kwargs = kwargs
        self.arguments = []
        self.mutually_exclusive_groups = []

    def add_argument(self, *args, **kwargs):
        arg = argument(*args, **kwargs)
        arg.parent = self
        self.arguments.append(arg)

    def add_mutually_exclusive_group(self, *args, **kwargs):
        group = mutually_exclusive_group(*args, **kwargs)
        group.parent = self
        self.mutually_exclusive_groups.append(group)
        return group

    def register(self, root):
        group = root.add_argument_group(*self.args, **self.kwargs)
        for x in self.arguments:
            x.register(group)
        for x in self.mutually_exclusive_groups:
            x.register(group)


class mutually_exclusive_group(object):
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


class Command(object):
    __metaclass__ = abc.ABCMeta

    @classmethod
    def _get_name(cls):
        if hasattr(cls, 'name'):
            return cls.name
        return cls.__name__.lower()

    @abc.abstractproperty
    def description(self):
        pass


class App(object):
    subcommand_dest = 'subcommand'

    def __init__(self, name, description, argv=None):
        self.name = name
        self.description = description
        if argv is None:
            argv = sys.argv[1:]
        if len(argv) == 0:
            argv.append('-h')
        self.parser = self._build_parser()
        self.args = self.parser.parse_args(argv)

    def start(self):
        name = getattr(self.args, self.subcommand_dest)
        exclude = [self.subcommand_dest]

        klass = [k for k in self._subcommands() if k._get_name() == name][0]
        dct = ((k, v) for k, v in self.args.__dict__.items() if k not in exclude)
        args = argparse.Namespace(**dict(dct))
        instance = klass(args)
        instance.start()
        return instance

    def _build_parser(self):
        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(dest=self.subcommand_dest)
        for klass in self._subcommands():
            subparser = subparsers.add_parser(
                klass._get_name(), description=klass.description)
            for v in klass.__dict__.values():
                if isinstance(v, (argument, argument_group, mutually_exclusive_group)):
                    if v.parent is None:
                        v.register(subparser)

        return parser

    @classmethod
    def _subcommands(cls):
        for subclass in Command.__subclasses__():
            if subclass != cls:
                yield subclass