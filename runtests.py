#!/usr/bin/env python
"""
runtests.py [OPTIONS] [-- ARGS]

Run tests, building the project first.

Examples::

    $ python runtests.py
    $ python runtests.py -t {SAMPLE_TEST}
"""
from __future__ import division, print_function

PROJECT_MODULE = "msmbuilder"
PROJECT_ROOT_FILES = ['msmbuilder', 'LICENSE', 'setup.py']
SAMPLE_TEST = "msmbuilder.tests.test_msm:test_ergodic_cutoff"

EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
              '/usr/local/lib/ccache', '/usr/local/lib/f90cache']

# ---------------------------------------------------------------------


if __doc__ is None:
    __doc__ = "Run without -OO if you want usage info"
else:
    __doc__ = __doc__.format(**globals())

import sys
import os

# In case we are run from the source directory, we don't want to import the
# project from there:
sys.path.pop(0)

import shutil
import subprocess
import time
from argparse import ArgumentParser, REMAINDER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("--no-build", "-n", action="store_true", default=False,
                        help="do not build the project "
                             "(use system installed version)")
    parser.add_argument("--build-only", "-b", action="store_true",
                        default=False,
                        help="just build, do not run any tests")
    parser.add_argument("--tests", "-t", action='append',
                        help="Specify tests to run")
    parser.add_argument("--debug", "-g", action="store_true",
                        help="Debug build")
    parser.add_argument("--show-build-log", action="store_true",
                        help="Show build output rather than using a log file")
    parser.add_argument("--verbose", "-v", action="count", default=1,
                        help="more verbosity")
    parser.add_argument("--no-verbose", action='store_true', default=False,
                        help="Default nose verbosity is -v. "
                             "This turns that off")
    parser.add_argument("--ipython", action='store_true', default=False,
                        help="Launch an ipython shell instead of nose")
    parser.add_argument("args", metavar="ARGS", default=[], nargs=REMAINDER,
                        help="Arguments to pass to Nose")
    args = parser.parse_args(argv)

    if not args.no_build:
        site_dir = build_project(args)
        sys.path.insert(0, site_dir)
        os.environ['PYTHONPATH'] = site_dir

    if args.build_only:
        sys.exit(0)

    if args.ipython:
        commands = ['ipython']
    else:
        commands = ['nosetests']

        if args.verbose > 0 and not args.no_verbose:
            verbosity = "-{vs}".format(vs="v" * args.verbose)
            commands += [verbosity]

        if args.tests:
            commands += args.tests[:]
        else:
            commands += [PROJECT_MODULE]

        extra_argv = args.args[:]
        if extra_argv and extra_argv[0] == '--':
            extra_argv = extra_argv[1:]
        commands += extra_argv

    # Run the tests under build/test
    test_dir = os.path.join("build", "test")
    try:
        shutil.rmtree(test_dir)
    except OSError:
        pass
    try:
        os.makedirs(test_dir)
    except OSError:
        pass

    cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        result = subprocess.call(commands)
    finally:
        os.chdir(cwd)

    sys.exit(result)


def build_project(args):
    """
    Build a dev version of the project.

    Returns
    -------
    site_dir
        site-packages directory where it was installed

    """

    root_ok = [os.path.exists(os.path.join(ROOT_DIR, fn))
               for fn in PROJECT_ROOT_FILES]
    if not all(root_ok):
        print("To build the project, run runtests.py in "
              "git checkout or unpacked source")
        sys.exit(1)

    dst_dir = os.path.join(ROOT_DIR, 'build', 'testenv')

    env = dict(os.environ)
    cmd = [sys.executable, 'setup.py']

    # Always use ccache, if installed
    prev_path = env.get('PATH', '').split(os.pathsep)
    env['PATH'] = os.pathsep.join(EXTRA_PATH + prev_path)

    if args.debug:
        # assume everyone uses gcc/gfortran
        env['OPT'] = '-O0 -ggdb'
        env['FOPT'] = '-O0 -ggdb'

    cmd += ["build"]
    # Install; avoid producing eggs so numpy can be imported from dst_dir.
    cmd += ['install', '--prefix=' + dst_dir,
            '--single-version-externally-managed',
            '--record=' + dst_dir + 'tmp_install_log.txt']

    log_filename = os.path.join(ROOT_DIR, 'build.log')

    if args.show_build_log:
        ret = subprocess.call(cmd, env=env, cwd=ROOT_DIR)
    else:
        log_filename = os.path.join(ROOT_DIR, 'build.log')
        print("Building, see build.log...")
        with open(log_filename, 'w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                 cwd=ROOT_DIR)

        # Wait for it to finish, and print something to indicate the
        # process is alive, but only if the log file has grown (to
        # allow continuous integration environments kill a hanging
        # process accurately if it produces no output)
        last_blip = time.time()
        last_log_size = os.stat(log_filename).st_size
        while p.poll() is None:
            time.sleep(0.5)
            if time.time() - last_blip > 60:
                log_size = os.stat(log_filename).st_size
                if log_size > last_log_size:
                    print("    ... build in progress")
                    last_blip = time.time()
                    last_log_size = log_size

        ret = p.wait()

    if ret == 0:
        print("Build OK")
    else:
        if not args.show_build_log:
            with open(log_filename, 'r') as f:
                print(f.read())
            print("Build failed!")
        sys.exit(1)

    from distutils.sysconfig import get_python_lib
    return get_python_lib(prefix=dst_dir, plat_specific=True)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
