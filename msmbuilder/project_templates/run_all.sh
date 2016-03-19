#!/usr/bin/env bash
set -e # exit on error

python -c 'import sys; assert sys.version_info >= (3,5), \
    "These scripts have been written for python 3.5. You have been warned!"' || true

python 0-get-example-data.py

python 1-gather-metadata.py
python 1-gather-metadata-plot.py

python 2-featurize.py
python 2-featurize-plot.py

python 2-rough-cluster.py
python 2-rough-cluster-plot.py