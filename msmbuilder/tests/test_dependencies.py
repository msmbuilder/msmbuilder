import os, pip, sys, warnings
from msmbuilder.example_datasets import has_msmb_data

def test_installed_packages():
    installed_packages = pip.get_installed_distributions()
    package_names = [package.project_name for package in installed_packages]

    test_dependencies = ['munkres', 'numdifftools', 'statsmodels', 'hmmlearn']

    if not hasattr(sys, 'getwindowsversion'):
        test_dependencies += ['cvxpy']

    for td in test_dependencies:
        if td not in package_names:
            raise RuntimeError('Please install {} to continue'.format(td))

def test_msmb_data():
    if has_msmb_data() is None:
        raise RuntimeError('Please install {} to continue'.format('msmb_data'))
