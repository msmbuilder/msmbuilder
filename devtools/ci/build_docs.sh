#!/usr/bin/env sh

# Print each line, exit on error
set -ev

# Install the built package
conda install --yes --use-local msmbuilder-dev

# TODO: Make this a conda package, include in requirements.txt
pip install msmb_theme

# Install doc requirements
conda install --yes --file doc/requirements.txt

cd doc
make html
