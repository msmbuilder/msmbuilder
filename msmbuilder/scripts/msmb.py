"""Statistical models for biomolecular dynamics"""
from __future__ import print_function, absolute_import, division
import sys
from ..cmdline import App
from ..commands import *
from ..version import version
# the commands register themselves when they're imported


class MSMBuilderApp(App):
    def _subcommands(self):
        cmds = super(MSMBuilderApp, self)._subcommands()

        # sort the commands in some arbitrary order.
        return sorted(cmds, key=lambda e: ''.join(x.__name__ for x in e.mro()))


def main():
    try:
        app = MSMBuilderApp(name='MSMBuilder', description=__doc__)
        app.start()
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        message = """\
An unexpected error has occurred with MSMBuilder (version %s), please
consider sending the following traceback to MSMBuilder GitHub issue tracker at:
            https://github.com/msmbuilder/msmbuilder/issues
"""
        print(message % version, file=sys.stderr)
        raise  # as if we did not catch it

if __name__ == '__main__':
    main()
