from __future__ import print_function

import itertools
import os
import re
import subprocess

import msmbuilder.cluster
import msmbuilder.decomposition
import msmbuilder.example_datasets
import msmbuilder.featurizer
import msmbuilder.msm

from numpy.testing.decorators import skipif


def get_commands_from_helptext():
    raw = subprocess.check_output(['msmb', '-h'], universal_newlines=True)
    lines = [l.strip() for l in raw.splitlines()]
    commandlist_i = lines.index('commands:') + 1
    lines = lines[commandlist_i:]

    for line in lines:
        if len(line) == 0:
            continue

        # regex for roughly the following:
        # CommandName        A longer description.
        ma = re.match(r"^(\w*)\s*\s\s([\w\s\-\(\):]*\.?)$", line)
        if ma is not None:
            print("Feat:", ma.group(1), "Text:", ma.group(2))
            yield ma.group(1)
        elif ' ' in line or line.endswith('.'):
            # If the command name is too long, it forces the description
            # to start on the next line. This is a heuristic for if this
            # is a "description only" line
            print("Text:", line)
        else:
            # See above, this is a regex for a 'command only' line.
            # It regex tests for camelcase
            ma = re.match(r"([A-Z][a-z]*)+", line)
            if ma is not None:
                print("Feat:", line)
                yield (line)
            else:
                # Ideally, this shouldn't happen
                # Run this file as a script and it will print all the
                # debug statements.
                print("No match!", repr(line))


HELP_COMMANDS = list(get_commands_from_helptext())


def get_from_module(module, exclude=None, include=None):
    if exclude is None:
        exclude = []

    ds_re = re.compile(r'([A-Z]+[a-z]*)+$')
    for k in module.__dict__.keys():
        if ds_re.match(k) is not None and k not in exclude:
            yield module.__name__, k

    if include is not None:
        for k in include:
            yield module.__name__, k


def get_all():
    return itertools.chain(
        get_from_module(msmbuilder.featurizer,
                        ['Featurizer',  # Base class
                         'SinMixin', 
                         'CosMixin',
                         'PsiMixin',
                         'PhiMixin',
                         'TransformerMixin',
                         'BaseEstimator',
                         'Parallel',
                         'BaseSubsetFeaturizer',  # Base class
                         'TrajFeatureUnion',
                         'Slicer',
                         'FirstSlicer',
                         'FeatureUnion',
                         'FunctionFeaturizer',
                         ]),
        get_from_module(msmbuilder.example_datasets,
                        ['MinimalFsPeptide',
                         ]),
        get_from_module(msmbuilder.cluster,
                        ['MultiSequenceClusterMixin',
                         'BaseEstimator',
                         ]),
        get_from_module(msmbuilder.decomposition,
                        exclude=['MultiSequenceDecompositionMixin',
                                 ],
                        include=['tICA']),
        get_from_module(msmbuilder.msm, )
    )


def assert_call(args):
    with open(os.devnull, 'w') as noout:
        assert subprocess.call(args,
                               stderr=subprocess.STDOUT,
                               stdout=noout) == 0


class CheckCommandHelpWorks(object):
    def __init__(self, modname, command):
        self.command = command
        self.description = "test_{}.{}_help_works".format(modname, command)

    def __call__(self):
        prefix = ['msmb']
        postfix = ['-h']
        assert_call(prefix + [self.command] + postfix)


class CheckCommandListed(object):
    def __init__(self, modname, command):
        self.command = command
        self.description = "test_{}.{}_listed".format(modname, command)

    def __call__(self):
        err = '{} was not listed in `msmb -h`'.format(self.command)
        assert self.command in HELP_COMMANDS, err

@skipif(True) # takes a long time
def test_all_help_works():
    for modname, feat in get_all():
        yield CheckCommandHelpWorks(modname, feat)


def test_all_listed():
    for modname, feat in get_all():
        yield CheckCommandListed(modname, feat)
