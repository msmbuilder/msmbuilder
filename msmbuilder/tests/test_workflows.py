from __future__ import print_function, division, absolute_import

import subprocess
from pkg_resources import resource_filename
from msmbuilder.tests.test_commands import tempdir


class CalledProcessError(Exception):
    def __init__(self, returncode, cmd, stdout=None, stderr=None):
         self.returncode = returncode
         self.cmd = cmd
         self.stdout = stdout
         self.stderr = stderr

    def __str__(self):
        return ("Command '%s' returned non-zero exit status %d.\n"
                "\n==== stdout ====\n%s\n==== stderr ====\n%s\n") % (self.cmd, self.returncode,
                                                 self.stdout, self.stderr)


def check_call(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate()
    except:
        process.kill()
        process.wait()
        raise

    retcode = process.poll()
    if retcode:
        raise CalledProcessError(retcode, cmd, stdout=stdout, stderr=stderr)


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
            check_call(line.split())


def test_workflow_2():
    with tempdir():
        for line in shell_lines('tests/workflows/test_2.sh'):
            check_call(line.split())


def test_workflow_3():
    with tempdir():
        for line in shell_lines('tests/workflows/test_3.sh'):
            check_call(line.split())
