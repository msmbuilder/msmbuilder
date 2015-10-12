conda install --yes `conda build devtools/conda-recipe --output`
pip install msmb_theme

# Install doc requirements
conda install --yes `cat doc/requirements.txt | xargs`

cd doc && make html && cd -
cat /tmp/sphinx-*

# TODO
# python devtools/ci/update-versions.py
