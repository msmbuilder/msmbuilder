from msmbuilder.io import TemplateProject
import tempfile
import shutil
import os


def setup_module():
    global WD, PWD
    PWD = os.path.abspath(".")
    WD = tempfile.mkdtemp()
    os.chdir(WD)


def teardown_module():
    os.chdir(PWD)
    shutil.rmtree(WD)


def test_template_project():
    tp = TemplateProject()
    tp.do()
