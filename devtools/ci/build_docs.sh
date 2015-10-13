#!/usr/bin/env sh

# Print each line, exit on error
set -ev

# Install the built package
conda create --yes -n docenv python=$CONDA_PY
source activate docenv
conda install --yes --use-local msmbuilder-dev

# TODO: Make this a conda package, include in requirements.txt
pip install msmb_theme

# Install doc requirements
conda install --yes --file doc/requirements.txt

# Make docs
cd doc && make html && cd -

# Move the docs into a versioned subdirectory
python devtools/ci/set_doc_version.py