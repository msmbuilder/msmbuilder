import os
import shutil

import msmbuilder.version

if msmbuilder.version.release:
    docversion = msmbuilder.version.short_version
else:
    docversion = 'development'

os.mkdir("doc/_deploy")
shutil.copytree("doc/_build/html", "doc/_deploy/{docversion}"
                .format(docversion=docversion))
