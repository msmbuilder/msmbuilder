"""mixtape: scikit-learn compatible mixture models and hidden Markov models

Currently, this package implements a mixture model of gamma distributions
and a hidden Markov model with von Mises emissions.

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
import glob
import numpy as np
from distutils.spawn import find_executable                     
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

###############################################################################
###############################################################################
def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            try:
                postargs = extra_postargs['gcc']
            except TypeError:
                postargs = []

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_executable('nvcc')
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)
###############################################################################
###############################################################################

def write_spline_data():
    """Precompute spline coefficients and save them to data files that
    are #included in the remaining c source code. This is a little devious.
    """
    import scipy.special
    import pyximport; pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    sys.path.insert(0, 'src/vonmises')
    import buildspline
    del sys.path[0]
    n_points = 1024
    miny, maxy = 1e-5, 700
    y = np.logspace(np.log10(miny), np.log10(maxy), n_points)
    x = scipy.special.iv(1, y) / scipy.special.iv(0, y)

    # fit the inverse function
    derivs = buildspline.createNaturalSpline(x, np.log(y))
    if not os.path.exists('src/vonmises/data/inv_mbessel_x.dat'):
        np.savetxt('src/vonmises/data/inv_mbessel_x.dat', x, newline=',\n')
    if not os.path.exists('src/vonmises/data/inv_mbessel_y.dat'):
        np.savetxt('src/vonmises/data/inv_mbessel_y.dat', np.log(y), newline=',\n')
    if not os.path.exists('src/vonmises/data/inv_mbessel_deriv.dat'):
        np.savetxt('src/vonmises/data/inv_mbessel_deriv.dat', derivs, newline=',\n')



_hmm = Extension('mixtape._hmm',
                 sources=['mixtape/_hmm.'+cython_extension],
                 libraries=['m'],
                 include_dirs=[np.get_include()])

_vmhmm = Extension('mixtape._vmhmm',
                   sources=['src/vonmises/vmhmm.c', 'src/vonmises/vmhmmwrap.'+cython_extension,
                            'src/vonmises/spleval.c',
                            'src/cephes/i0.c', 'src/cephes/chbevl.c'],
                   libraries=['m'],
                   include_dirs=[np.get_include(), 'src/cephes'])

_gamma = Extension('mixtape._gamma',
                      sources=['src/gamma/gammawrap.'+cython_extension,
                               'src/gamma/gammamixture.c', 'src/gamma/gammautils.c',
                               'src/cephes/zeta.c', 'src/cephes/psi.c', 'src/cephes/polevl.c',
                               'src/cephes/mtherr.c', 'src/cephes/gamma.c'],
                      libraries=['m'],
                      extra_compile_args=['--std=c99', '-Wall'],
                      include_dirs=[np.get_include(), 'src/cephes'])

_cudahmm = Extension('mixtape._cudahmm',
                     language="c++",
                     library_dirs=[CUDA['lib64']],
                     libraries=['cudart', 'cublas'],
                     runtime_library_dirs=[CUDA['lib64']],
                     extra_compile_args={'gcc': [],
                                         'nvcc': ['-arch=sm_30', '-G', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                     sources=['platforms/cuda/wrappers/CUDAGaussianHMM.pyx',
                              'platforms/cuda/src/CUDAGaussianHMM.cu'],
                     include_dirs=[np.get_include(), 'platforms/cuda/include', 'platforms/cuda/kernels'])

_cpuhmm = Extension('mixtape._cpuhmm',
                    language='c++',
                    sources=['platforms/cpu/wrappers/CPUGaussianHMM.pyx'] +
                              glob.glob('platforms/cpu/kernels/*.c'),
                    include_dirs=[np.get_include(),
                                  'platforms/cpu/kernels',
                                  'platforms/cpu/kernels/include/'])


write_spline_data()
setup(name='mixtape',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      url='https://github.com/rmcgibbo/mixtape',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),
      packages=['mixtape'],
      zip_safe=False,
      ext_modules=[_hmm, _vmhmm, _gamma, _cudahmm, _cpuhmm],
      cmdclass={'build_ext': custom_build_ext})
