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
import numpy as np
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

def write_spline_data():
    """Precompute spline coefficients and save them to data files that
    are #included in the remaining c source code. This is a little devious.
    """
    import scipy.special
    import pyximport; pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    sys.path.insert(0, 'src')
    import buildspline
    del sys.path[0]
    n_points = 1024
    miny, maxy = 1e-5, 700
    y = np.logspace(np.log10(miny), np.log10(maxy), n_points)
    x = scipy.special.iv(1, y) / scipy.special.iv(0, y)

    # fit the inverse function
    derivs = buildspline.createNaturalSpline(x, np.log(y))
    if not os.path.exists('src/data/inv_mbessel_x.dat'):
        np.savetxt('src/data/inv_mbessel_x.dat', x, newline=',\n')
    if not os.path.exists('src/data/inv_mbessel_y.dat'):
        np.savetxt('src/data/inv_mbessel_y.dat', np.log(y), newline=',\n')
    if not os.path.exists('src/data/inv_mbessel_deriv.dat'):
        np.savetxt('src/data/inv_mbessel_deriv.dat', derivs, newline=',\n')


_vmhmm = Extension('_vmhmm',
                   sources=['src/vmhmm.c', 'src/vmhmmwrap.'+cython_extension,
                            'src/spleval.c',
                            'src/cephes/i0.c', 'src/cephes/chbevl.c'],
                   libraries=['m'],
                   include_dirs=[np.get_include(), 'src/cephes'])
_gammahmm = Extension('_gammahmm',
                      sources=['src/gammahmmwrap.'+cython_extension,
                               'src/cephes/zeta.c', 'src/cephes/psi.c', 'src/cephes/polevl.c',
                               'src/cephes/mtherr.c', 'src/cephes/gamma.c',
                               'src/gammamixture.c', 'src/gammautils.c'],
                      libraries=['m'],
                      extra_compile_args=['--std=c99', '-Wall'],
                      include_dirs=[np.get_include(), 'src/cephes'])

write_spline_data()
setup(name='vmhmm',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      url='https://github.com/rmcgibbo/vmhmm',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),      
      py_modules=['vmhmm'],
      zip_safe=False,
      ext_modules=[_vmhmm, _gammahmm], **setup_kwargs)
