sudo apt-get install -qq -y g++ gfortran valgrind csh
sudo apt-get install -qq -y g++-multilib gcc-multilib
wget http://repo.continuum.io/miniconda/Miniconda-3.0.5-Linux-x86_64.sh
bash Miniconda-3.0.5-Linux-x86_64.sh -b

export PATH=$HOME/miniconda/bin:$PATH

conda config --add channels http://conda.binstar.org/omnia
conda create --yes -n ${python} python=${python} numpy scipy cython \
    pandas pip netcdf4 scikit-learn cvxopt nose mdtraj
source activate $python

# PYTHON_VERSION=`python -c 'import sys; print("%d.%d" % sys.version_info[:2])'`
# pip install $PIP_ARGS -r tools/ci/requirements-${PYTHON_VERSION}.txt
