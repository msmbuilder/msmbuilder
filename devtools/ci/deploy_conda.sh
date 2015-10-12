conda install --yes anaconda-client
binstar upload -u omnia -t $BINSTAR_TOKEN --force `conda build devtools/conda-recipe --output`
