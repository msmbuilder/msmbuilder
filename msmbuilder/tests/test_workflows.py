from __future__ import print_function, division, absolute_import

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


def test_workflow_1():
    with tempdir():
        for line in shell_lines('tests/workflows/test_1.sh'):
            subprocess.check_call(line.split())


def test_workflow_2():
    with tempdir():
        for line in shell_lines('tests/workflows/test_2.sh'):
            subprocess.check_call(line.split())


def test_workflow_3():
    with tempdir():
        for line in shell_lines('tests/workflows/test_3.sh'):
            subprocess.check_call(line.split())
