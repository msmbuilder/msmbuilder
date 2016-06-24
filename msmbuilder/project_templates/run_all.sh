#!/usr/bin/env bash
set -e # exit on error

python 0-test-install.py
python 0-get-example-data.py

python 1-gather-metadata.py
python 1-gather-metadata-plot.py

python 2-featurize.py
python 2-featurize-plot.py

python 2-rough-cluster.py
python 2-rough-cluster-plot.py