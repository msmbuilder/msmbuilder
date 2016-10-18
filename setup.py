"""MSMBuilder: Statistical models for Biomolecular Dynamics
"""

from __future__ import print_function, absolute_import

DOCLINES = __doc__.split("\n")

import sys
import traceback
import numpy as np
from os.path import join as pjoin
from setuptools import setup, Extension, find_packages

try:
    sys.dont_write_bytecode = True
    sys.path.insert(0, '.')
    from basesetup import write_version_py, CompilerDetection, \
        check_dependencies
finally:
    sys.dont_write_bytecode = False

try:
    import mdtraj

    mdtraj_capi = mdtraj.capi()
except (ImportError, AttributeError):
    print('=' * 80)
    print('MDTraj version 1.1.X or later is required')
    print('=' * 80)
    traceback.print_exc()
    sys.exit(1)

if '--debug' in sys.argv:
    sys.argv.remove('--debug')
    DEBUG = True
else:
    DEBUG = False
if '--disable-openmp' in sys.argv:
    sys.argv.remove('--disable-openmp')
    DISABLE_OPENMP = True
else:
    DISABLE_OPENMP = False

try:
    import Cython
    from Cython.Distutils import build_ext

    if Cython.__version__ < '0.18':
        raise ImportError()
except ImportError:
    print(
        'Cython version 0.18 or later is required. Try "conda install cython"')
    sys.exit(1)

# #########################
VERSION = '3.6.1'
ISRELEASED = True
__version__ = VERSION
# #########################

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)
Programming Language :: C++
Programming Language :: Python
Development Status :: 5 - Production/Stable
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
"""

if any(cmd in sys.argv for cmd in ('install', 'build', 'develop')):
    check_dependencies((
        ('numpy',),
        ('scipy',),
        ('pandas',),
        ('six',),
        ('mdtraj',),
        ('sklearn', 'scikit-learn'),
        ('numpydoc',),
        ('tables', 'pytables'),
    ))

# Where to find extensions
MSMDIR = 'msmbuilder/msm/'
HMMDIR = 'msmbuilder/hmm/'
CLUSTERDIR = 'msmbuilder/cluster/'

compiler = CompilerDetection(DISABLE_OPENMP)
with open('msmbuilder/src/config.pxi', 'w') as f:
    f.write('''
DEF DEBUG = {debug}
DEF OPENMP = {openmp}
    '''.format(openmp=compiler.openmp_enabled, debug=DEBUG))

extensions = []
extensions.append(
    Extension('msmbuilder.example_datasets._muller',
              sources=[pjoin('msmbuilder', 'example_datasets', '_muller.pyx')],
              include_dirs=[np.get_include()]))

extensions.append(
    Extension('msmbuilder.msm._markovstatemodel',
              sources=[pjoin(MSMDIR, '_markovstatemodel.pyx'),
                       pjoin(MSMDIR, 'src/transmat_mle_prinz.c')],
              include_dirs=[pjoin(MSMDIR, 'src'), np.get_include()]))

extensions.append(
    Extension('msmbuilder.tests.test_cyblas',
              sources=['msmbuilder/tests/test_cyblas.pyx'],
              include_dirs=['msmbuilder/src', np.get_include()]))

extensions.append(
    Extension('msmbuilder.msm._ratematrix',
              sources=[pjoin(MSMDIR, '_ratematrix.pyx')],
              language='c++',
              extra_compile_args=compiler.compiler_args_openmp,
              libraries=compiler.compiler_libraries_openmp,
              include_dirs=['msmbuilder/src', np.get_include()]))

extensions.append(
    Extension('msmbuilder.decomposition._speigh',
              sources=[pjoin('msmbuilder', 'decomposition', '_speigh.pyx')],
              language='c++',
              extra_compile_args=compiler.compiler_args_openmp,
              libraries=compiler.compiler_libraries_openmp,
              include_dirs=['msmbuilder/src', np.get_include()]))

extensions.append(
    Extension('msmbuilder.msm._metzner_mcmc_fast',
              sources=[pjoin(MSMDIR, '_metzner_mcmc_fast.pyx'),
                       pjoin(MSMDIR, 'src/metzner_mcmc.c')],
              libraries=compiler.compiler_libraries_openmp,
              extra_compile_args=compiler.compiler_args_openmp,
              include_dirs=[pjoin(MSMDIR, 'src'), np.get_include()]))

extensions.append(
    Extension('msmbuilder.libdistance',
              language='c++',
              sources=['msmbuilder/libdistance/libdistance.pyx'],
              # msvc needs to be told "libtheobald", gcc wants just "theobald"
              libraries=['%stheobald' % ('lib' if compiler.msvc else '')],
              include_dirs=["msmbuilder/libdistance/src",
                            mdtraj_capi['include_dir'], np.get_include()],
              library_dirs=[mdtraj_capi['lib_dir']],
              ))

extensions.append(
    Extension('msmbuilder.cluster._kmedoids',
              language='c++',
              sources=[pjoin(CLUSTERDIR, '_kmedoids.pyx'),
                       pjoin(CLUSTERDIR, 'src', 'kmedoids.cc')],
              include_dirs=[np.get_include()]))

# To get debug symbols on Windows, use
# extra_link_args=['/DEBUG']
# extra_compile_args=['/Zi']

extensions.append(
    Extension('msmbuilder.hmm.gaussian',
              language='c++',
              sources=[pjoin(HMMDIR, 'gaussian.pyx'),
                       pjoin(HMMDIR, 'src/GaussianHMMFitter.cpp')],
              libraries=compiler.compiler_libraries_openmp,
              extra_compile_args=compiler.compiler_args_sse3
                                 + compiler.compiler_args_openmp,
              include_dirs=[np.get_include(),
                            HMMDIR,
                            pjoin(HMMDIR, 'src/include/'),
                            pjoin(HMMDIR, 'src/')]))

extensions.append(
    Extension('msmbuilder.hmm.vonmises',
              language='c++',
              sources=[pjoin(HMMDIR, 'vonmises.pyx'),
                       pjoin(HMMDIR, 'src/VonMisesHMMFitter.cpp'),
                       pjoin(HMMDIR, 'cephes/i0.c'),
                       pjoin(HMMDIR, 'cephes/chbevl.c')],
              libraries=compiler.compiler_libraries_openmp,
              extra_compile_args=compiler.compiler_args_sse3
                                 + compiler.compiler_args_openmp,
              include_dirs=[np.get_include(),
                            HMMDIR,
                            pjoin(HMMDIR, 'src/include/'),
                            pjoin(HMMDIR, 'src/'),
                            pjoin(HMMDIR, 'cephes/')]))

write_version_py(VERSION, ISRELEASED, filename='msmbuilder/version.py')
setup(name='msmbuilder',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      url='https://github.com/msmbuilder/msmbuilder',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),
      packages=find_packages(),
      package_data={
          'msmbuilder.tests': ['workflows/*'],
          'msmbuilder': ['project_templates/*.*',
                         'project_templates/*/*',
                         'io_templates/*',
                         ],
      },
      entry_points={'console_scripts':
                        ['msmb = msmbuilder.scripts.msmb:main']},
      zip_safe=False,
      ext_modules=extensions,
      cmdclass={'build_ext': build_ext})
