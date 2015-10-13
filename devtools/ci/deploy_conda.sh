#!/bin/bash

conda install --yes anaconda-client
anaconda -t $BINSTAR_TOKEN upload -u omnia --force `conda build devtools/conda-recipe --output`
