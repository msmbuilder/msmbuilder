"""This script tests your python installation as it pertains to running project templates.

MSMBuilder supports Python 2.7 and 3.3+ and has some necessary dependencies
like numpy, scipy, and scikit-learn. This templated project enforces
some more stringent requirements to make sure all the users are more-or-less
on the same page and to allow developers to exploit more helper libraries.

You can modify the template scripts to work for your particular set-up,
but it's probably easier to install `conda` and get the packages we
recommend.

{{header}}
"""

import textwrap

# Show intro text
paragraphs = __doc__.split('\n\n')
for p in paragraphs:
    print(textwrap.fill(p))
    print()

warnings = 0

## Test for python 3.5
import sys

if sys.version_info < (3, 5):
    print(textwrap.fill(
        "These scripts were all developed on Python 3.5, "
        "which is the current, stable release of Python. "
        "In particular, we use subprocess.run "
        "(and probably some other new features). "
        "You can easily modify the scripts to work on older versions "
        "of Python, but why not just upgrade? We like Continuum's "
        "Anaconda Python distribution for a simple install (without root)."
    ))
    print()
    warnings += 1

## Test for matplotlib
try:
    import matplotlib as plt
except ImportError:
    print(textwrap.fill(
        "These scripts try to make some mildly intesting plots. "
        "That requires `matplotlib`."
    ))
    print()
    warnings += 1

## Test for seaborn
try:
    import seaborn as sns
except ImportError:
    print(textwrap.fill(
        "The default matplotlib styling is a little ugly. "
        "By default, these scripts try to use `seaborn` to make prettier "
        "plots. You can remove all the seaborn imports if you don't want "
        "to install this library, but why not just install it? Try "
        "`conda install seaborn`"
    ))
    print()
    warnings += 1

## Test for xdg-open
try:
    import subprocess

    subprocess.check_call(['xdg-open', '--version'])
except:
    print(textwrap.fill(
        "For convenience, the plotting scripts can try to use `xdg-open` "
        "to pop up the result of the plot. Use the --display flag on "
        "msmb TemplateProject to enable this behavior."
    ))
    warnings += 1

## Report results
if warnings == 0:
    print("I didn't find any problems with your installation! Good job.")
    print()
else:
    print("I found {} warnings, see above. Good luck!".format(warnings))
    print()
