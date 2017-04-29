import pip
import os
import warnings

def test_installed_packages():
    installed_packages = pip.get_installed_distributions()
    package_names = [package.project_name for package in installed_packages]

    test_dependencies = ['munkres', 'numdifftools', 'statsmodels', 'hmmlearn', 'cvxpy']

    for td in test_dependencies:
        if td not in package_names:
            raise RuntimeError('Please install {} to continue'.format(td))

def test_msmbuilder_data():
    home = os.path.expanduser('~')
    try:
        os.stat(home+'/msmbuilder_data/')
    except:
        raise RuntimeError('Please conda install msmb_data')

    try:
        assert len([d for d in os.listdir(home+'/msmbuilder_data/')
                    if os.path.isdir(home+'/msmbuilder_data/'+d)]) >= 6
    except:
        warnings.warn('You may not have all the required msmbuilder_data ' +
                      'subdirectories. Try running conda install msmb_data.')
