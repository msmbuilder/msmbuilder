import sys
import abc
import argparse

__all__ = ['argument', 'Command', 'App']

class argument(object):
    """Wrapper for arguments that will get passed to
    argparse.ArgumentParser.add_argument when a subcommand's
    parser is created. For each argument that a subcommand
    wants, it should add one `argument` class attribute
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Command(object):
    """Superclass for commands
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def _get_name(cls):
        if hasattr(cls, 'name'):
            return cls.name
        return cls.__name__.lower()

    @abc.abstractproperty
    def description(self):
        pass


class _RootCommand(Command):
    description = None

    @classmethod
    def _subcommands(cls):
        for subclass in Command.__subclasses__():
            if subclass != cls:
                yield subclass

    def __init__(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        if len(argv) == 1:
            argv.append('-h')
        self.parser = self._build_parser()
        self.args = self.parser.parse_args(argv)

    def start(self):
        cls = None
        for klass in self._subcommands():
            if klass._get_name() == self.args.__subparser__name__:
                cls = klass()
        if cls is None:
            raise RuntimeError()
      
        args = argparse.Namespace(**dict(((k, v) for k, v in self.args.__dict__.items() if k != '__subparser__name__')))
        cls.start(args)

    def _build_parser(self):
        parser = argparse.ArgumentParser(description=self.description,
                                         usage='%s [subcommand]' % self._get_name())
        subparsers = parser.add_subparsers(dest='__subparser__name__')
        for klass in self._subcommands():
            subparser = subparsers.add_parser(klass._get_name(),
                                              help=klass.description)
            for k, v in klass.__dict__.items():
                if isinstance(v, argument):
                    if 'dest' in v.kwargs:
                        raise ValueError('dest is not supported')
                    subparser.add_argument(*v.args, dest=k, **v.kwargs)
        return parser


def App(name, description):
    rc = _RootCommand()
    rc.name = name
    rc.description = description
    return rc
