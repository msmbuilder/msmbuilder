import os
import shutil

import msmbuilder.version

if msmbuilder.version.release:
    docversion = msmbuilder.version.short_version
else:
    docversion = 'latest'

os.mkdir("doc/_deploy")
shutil.copytree("doc/_build", "doc/_deploy/{docversion}"
                .format(docversion=docversion))
