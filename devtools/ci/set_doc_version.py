import os
import shutil
from msmbuilder import version

if version.release:
    docversion = version.short_version
else:
    docversion = 'development'

os.mkdir("doc/_deploy")
shutil.copytree("doc/_build/html", "doc/_deploy/{docversion}"
                .format(docversion=docversion))
