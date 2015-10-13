#!/bin/bash
conda install --yes anaconda-client
anaconda upload -u omnia -t $BINSTAR_TOKEN --force `conda build devtools/conda-recipe --output`
