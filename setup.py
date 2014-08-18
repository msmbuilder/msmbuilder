"""mixtape: hidden Markov models and beyond
"""

from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import glob
import copy
import shutil
import textwrap
import tempfile
import subprocess
from distutils.ccompiler import new_compiler
from distutils.spawn import find_executable
from setuptools import setup, Extension, find_packages

import numpy as np
from numpy.distutils import system_info


try:
    import Cython
    from Cython.Distutils import build_ext
    if Cython.__version__ < '0.18':
        raise ImportError()
except ImportError:
    print('Cython version 0.18 or later is required. Try "easy_install cython"')
    sys.exit(1)

##########################
VERSION = '0.2'
ISRELEASED = False
__version__ = VERSION
##########################

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)
Programming Language :: C++
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

################################################################################
# Writing version control information to the module
################################################################################

def git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def write_version_py(filename='Mixtape/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM MIXTAPE SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


###############################################################################
# CUDA stuff
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
    #default_compiler_so[0] = 'g++'
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
                postargs = extra_postargs

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def get_lapack():
    from collections import defaultdict
    lapack_info = defaultdict(lambda: [])
    lapack_info.update(system_info.get_info('lapack'))
    if len(lapack_info) == 0:
        try:
            from scipy.linalg import _flapack
            lapack_info['extra_link_args'] = [_flapack.__file__]
            return lapack_info
        except ImportError:
            pass
        print('LAPACK libraries could not be located.', file=sys.stderr)
        sys.exit(1)
    return lapack_info


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
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_executable('nvcc')
        if nvcc is None:
            raise EnvironmentError(
                'The nvcc compiler could not be located in your $PATH. '
                'To enable CUDA acceleration, either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


class custom_build_ext(build_ext):
    def build_extensions(self):
        # Add the NVCC hacks
        customize_compiler_for_nvcc(self.compiler)

        # Here come the cython hacks
        from distutils.command.build_ext import build_ext as _build_ext
        # first, AVOID calling cython.build_ext's build_extensions
        # method, because it cythonizes all of the pyx files to cpp
        # here, which we do *not* want to do. Instead, we want to do
        # them one at a time during build_extension so that we can
        # regenerate them on every extension. This is necessary for getting
        # the single/mixed precision builds to work correctly, because we
        # use the same pyx file, with different macros, and make differently
        # named extensions. Since each extension needs to have a unique init
        # method in the cpp code, the cpp needs to be translated fresh from
        # pyx.
        _build_ext.build_extensions(self)

    def build_extension(self, ext):
        build_ext.cython_gdb = True
        # Clean all cython files for each extension
        # and force the cpp files to be rebuilt from pyx.
        cplus = self.cython_cplus or getattr(ext, 'cython_cplus', 0) or \
                (ext.language and ext.language.lower() == 'c++')
        if len(ext.define_macros) > 0:
            for f in ext.sources:
                if f.endswith('.pyx'):
                    if cplus:
                        compiled = f[:-4] + '.cpp'
                    else:
                        compiled = f[:-4] + '.c'
                    if os.path.exists(compiled):
                        os.unlink(compiled)
        ext.sources = self.cython_sources(ext.sources, ext)
        build_ext.build_extension(self, ext)



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


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    "Does this compiler support OpenMP parallelization?"
    compiler = new_compiler()
    print('\n\033[95mAttempting to autodetect OpenMP support...\033[0m')
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print('\033[92mCompiler supports OpenMP\033[0m\n')
    else:
        print('\033[91mDid not detect OpenMP support; parallel support disabled\033[0m\n')
    return hasopenmp, needs_gomp


openmp_enabled, needs_gomp = detect_openmp()
extra_compile_args = ['-msse3']
if openmp_enabled:
    extra_compile_args.append('-fopenmp')
libraries = ['gomp'] if needs_gomp else []
extensions = []
lapack_info = get_lapack()

extensions.append(
    Extension('mixtape._reversibility',
              sources=['src/reversibility.pyx'],
              libraries=['m'],
              include_dirs=[np.get_include()]))

extensions.append(
    Extension('mixtape.datasets._muller',
              sources=['Mixtape/datasets/_muller.pyx'],
              libraries=['m'],
              include_dirs=[np.get_include()]))

extensions.append(
    Extension('mixtape.markovstatemodel._markovstatemodel',
              sources=['Mixtape/markovstatemodel/_markovstatemodel.pyx',
                       'Mixtape/markovstatemodel/src/transmat_mle_prinz.c'],
              libraries=['m'],
              include_dirs=['Mixtape/markovstatemodel/src', np.get_include()]))

extensions.append(
    Extension('mixtape.markovstatemodel._metzner_mcmc_fast',
              sources=['Mixtape/markovstatemodel/_metzner_mcmc_fast.pyx',
                       'Mixtape/markovstatemodel/src/metzner_mcmc.c'],
              libraries=['m'] + libraries,
              extra_compile_args=extra_compile_args,
              include_dirs=['Mixtape/markovstatemodel/src', np.get_include()]))

extensions.append(
    Extension('mixtape.cluster._regularspatialc',
              sources=['Mixtape/cluster/_regularspatialc.pyx'],
              libraries=['m'],
              include_dirs=['Mixtape/src/f2py', 'Mixtape/src/blas', np.get_include()]))

extensions.append(
    Extension('mixtape.cluster._kcentersc',
              sources=['Mixtape/cluster/_kcentersc.pyx'],
              libraries=['m'],
              include_dirs=['Mixtape/src/f2py', 'Mixtape/src/blas', np.get_include()]))

extensions.append(
    Extension('mixtape.cluster._commonc',
              sources=['Mixtape/cluster/_commonc.pyx'],
              libraries=['m'],
              include_dirs=['Mixtape/src/f2py', 'Mixtape/src/blas', np.get_include()]))

extensions.append(
    Extension('mixtape._ghmm',
              language='c++',
              sources=['platforms/cpu/wrappers/GaussianHMMCPUImpl.pyx'] +
                        glob.glob('platforms/cpu/kernels/*.c') +
                        glob.glob('platforms/cpu/kernels/*.cpp'),
              libraries=libraries + lapack_info['libraries'],
              extra_compile_args=extra_compile_args,
              extra_link_args=lapack_info['extra_link_args'],
              include_dirs=[np.get_include(), 'platforms/cpu/kernels/include/',
                            'platforms/cpu/kernels/']))

extensions.append(
    Extension('mixtape._mslds',
              language='c++',
              sources=['platforms/cpu/wrappers/MetastableSLDSCPUImpl.pyx'] +
                        glob.glob('platforms/cpu/kernels/*.c') +
                        glob.glob('platforms/cpu/kernels/*.cpp'),
              libraries=libraries + lapack_info['libraries'],
              extra_compile_args=extra_compile_args,
              extra_link_args=lapack_info['extra_link_args'],
              include_dirs=[np.get_include(), 'platforms/cpu/kernels/include/',
                            'platforms/cpu/kernels/']))

extensions.append(
    Extension('mixtape._vmhmm',
              sources=['src/vonmises/vmhmm.c', 'src/vonmises/vmhmmwrap.pyx',
                       'src/vonmises/spleval.c',
                       'src/cephes/i0.c', 'src/cephes/chbevl.c'],
              libraries=['m'],
              include_dirs=[np.get_include(), 'src/cephes']))

try:
    if '--disable-cuda' in sys.argv:
        sys.argv.remove('--disable-cuda')
        raise EnvironmentError()

    CUDA = locate_cuda()
    kwargs = dict(
        language="c++",
        library_dirs=[CUDA['lib64']],
        libraries=['cudart', 'cublas'],
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={'gcc': [],
                            'nvcc': ['-arch=sm_30', '-c', '--compiler-options', "'-fPIC'"]},
        sources=['platforms/cuda/wrappers/GaussianHMMCUDAImpl.pyx',
                 'platforms/cuda/src/CUDAGaussianHMM.cu'],
        include_dirs=[np.get_include(), 'platforms/cuda/include', 'platforms/cuda/kernels'])

    extensions.append(
        Extension('mixtape._cuda_ghmm_single', define_macros=[('mixed', 'float')], **kwargs))
    extensions.append(
        Extension('mixtape._cuda_ghmm_mixed', define_macros=[('mixed', 'double')], **kwargs))


except EnvironmentError as e:
    print('\033[91m%s' % '#'*60)
    print("\n".join(textwrap.wrap(str(e), 60)))
    print('#'*60, '\033[0m\n')


write_version_py()
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
      packages=['mixtape'] + ['mixtape.%s' % e for e in find_packages('Mixtape')],
      package_dir={'mixtape': 'Mixtape'},
      scripts=['scripts/hmsm', 'scripts/mixtape', 'scripts/pbsipcluster'],
      zip_safe=False,
      ext_modules=extensions,
#      install_requires=['IPython', 'scikit-learn>=0.14', 'six', 'numpydoc',
#                        'mdtraj>=0.8.0', 'scipy>=0.11.0',
#                        'pandas>=0.9.0', 'cvxopt>=1.1.5'],
      cmdclass={'build_ext': custom_build_ext})
