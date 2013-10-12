"""vmhmm: scikit-learn compatible von Mises hidden Markov model

This package implements a hidden Markov model with von Mises emissions.
See http://scikit-learn.org/stable/modules/hmm.html for a 
practical description of hidden Markov models. The von Mises
distribution, (also known as the circular normal distribution or
Tikhonov distribution) is a continuous probability distribution on
the circle. For multivariate signals, the emmissions distribution
implemented by this model is a product of univariate von Mises
distributuons -- analogous to the multivariate Gaussian distribution
with a diagonal covariance matrix.
"""

from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import numpy
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

try:
    from Cython.Distutils import build_ext
    setup_kwargs = {'cmdclass': {'build_ext': build_ext}}
    cython_extension = 'pyx'
except ImportError:
    setup_kwargs = {}
    cython_extension = 'c'

##########################
__version__ = 0.1
##########################

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C
Programming Language :: Python
Development Status :: 3 - Alpha
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
"""

_vmhmm = Extension('_vmhmm',
    sources=['src/_vmhmm.c', 'src/_vmhmmwrap.' + cython_extension],
    libraries=['m'],
    include_dirs=[numpy.get_include()])

setup(name='vmhmm',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      url='http://rmcgibbo.github.io/mdtraj',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),      
      py_modules=['vmhmm'],
      zip_safe=False,
      ext_modules=[_vmhmm], **setup_kwargs)
