"""Statistical models for biomolecular dynamics"""
from ..cmdline import App
from ..commands import *
# the commands register themselves when they're imported


class MSMBuilderApp(App):
    def _subcommands(self):
        cmds = super(MSMBuilderApp, self)._subcommands()

        # sort the commands in some arbitrary order.
        return sorted(cmds, key=lambda e: ''.join(x.__name__ for x in e.mro()))


def main():
    app = MSMBuilderApp(name='MSMBuilder', description=__doc__)
    app.start()


if __name__ == '__main__':
    main()
