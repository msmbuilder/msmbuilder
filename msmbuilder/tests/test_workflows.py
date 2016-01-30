from __future__ import print_function, division, absolute_import

import os
import subprocess

from pkg_resources import resource_filename

from msmbuilder.tests.test_commands import tempdir


def shell_lines(resource):
    fn = resource_filename('msmbuilder', resource)
    buf = ''
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith('\\'):
                buf += line.rstrip('\\')
            else:
                yield buf + ' ' + line
                buf = ''


def check_call(tokens):
    # ugh python 2 is the worst. newer pythons have subprocess.DEVNULL
    with open(os.devnull, 'w') as devnull:
        try:
            out = subprocess.check_output(tokens, stderr=subprocess.STDOUT,
                                          universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.cmd)
            print(e.output)
            raise


def test_workflow_1():
    with tempdir():
        for line in shell_lines('tests/workflows/test_1.sh'):
            check_call(line.split())


def test_workflow_2():
    with tempdir():
        for line in shell_lines('tests/workflows/test_2.sh'):
            check_call(line.split())


def test_workflow_3():
    with tempdir():
        for line in shell_lines('tests/workflows/test_3.sh'):
            check_call(line.split())
