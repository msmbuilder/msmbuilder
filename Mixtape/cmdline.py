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
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division
import re
import sys
import abc
import argparse
from IPython.utils.text import wrap_paragraphs

__all__ = ['argument', 'argument_group', 'Command', 'App', 'MultipleIntAction']

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
            parsed = map(int, re.findall('[^,;\s]+', values))
            setattr(namespace, self.dest, parsed)
        except ValueError:
            raise argparse.ArgumentError(self, 'Invalid list of integers: "%s"' % values)

#-----------------------------------------------------------------------------
# Argument Declaration Class Attibutes
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

    def add_argument(self, *args, **kwargs):
        arg = argument(*args, **kwargs)
        arg.parent = self
        self.children.append(arg)

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
        
    def error(self, msg):
        print(msg, file=sys.stderr)
        exit(1)


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
        # Using a custom "MyHelpFormatter" to monkey-patch argparse into making
        # all of the subcommands get rendered in the help text on one line. To
        # do this, you need to increase the "action_max_length" argument which
        # puts more whitespace between the end of the action name and the start
        # of the helptext.
        parser = argparse.ArgumentParser(description=self.description,
            formatter_class=lambda prog: MyHelpFormatter(prog,
            indent_increment=1, width=88, action_max_length=17))

        subparsers = parser.add_subparsers(dest=self.subcommand_dest, title="commands", metavar="")
        for klass in self._subcommands():
            # http://stackoverflow.com/a/17124446/1079728
            first_sentence = ' '.join(' '.join(re.split(r'(?<=[.:;])\s', klass.description)[:1]).split())
            description = '\n\n'.join(wrap_paragraphs(klass.description))
            subparser = subparsers.add_parser(
                klass._get_name(), help=first_sentence, description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
            for v in (getattr(klass, e) for e in dir(klass)):
                if isinstance(v, (argument, argument_group, mutually_exclusive_group)):
                    if v.parent is None:
                        v.register(subparser)

        return parser

    @classmethod
    def _subcommands(cls):
        for subclass in all_subclasses(Command):
            if subclass != cls:
                yield subclass

def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        action_max_length = kwargs.pop('action_max_length', 0)
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._action_max_length = action_max_length