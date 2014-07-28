# PYTHON_VERSION=`python -c 'import sys; print("%d.%d" % sys.version_info[:2])'`
# coveralls

echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" == "true" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi

if [[ `python -c "import sys; print(sys.version_info[:2])"` != "(2, 7)" ]]; then
    echo "No deploy on PYTHON_VERSION=${PYTHON_VERSION}"; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi

# Create the docs and push them to S3
# -----------------------------------

# Install stuff for running the example IPython notebooks
sudo apt-get install -qq pandoc         # notebook -> rst
conda install --yes matplotlib scikit-learn sphinx boto ipython-notebook jinja2
pip install runipy==0.0.4                      # example notebooks

cd doc && make html && cd -
cat /tmp/sphinx-*
python tools/ci/push-docs-to-s3.py
