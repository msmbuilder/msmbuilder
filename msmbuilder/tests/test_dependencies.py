import os, pip, sys, warnings

def test_installed_packages():
    installed_packages = pip.get_installed_distributions()
    package_names = [package.project_name for package in installed_packages]

    test_dependencies = ['munkres', 'numdifftools', 'statsmodels', 'hmmlearn']

    if not hasattr(sys, 'getwindowsversion'):
        test_dependencies += ['cvxpy']

    for td in test_dependencies:
        if td not in package_names:
            raise RuntimeError('Please install {} to continue'.format(td))


def test_fs_peptide():
    try:
        from msmbuilder.example_datasets import FsPeptide
        fspeptide = FsPeptide().get()
    except:
        warnings.warn('Fs peptide dataset not found. Did you install msmb_data?')


def test_alanine_dipeptide():
    try:
        from msmbuilder.example_datasets import AlanineDipeptide
        adpeptide = AlanineDipeptide().get()
    except:
        warnings.warn('Alanine dipeptide dataset not found. Did you install msmb_data?')


def test_double_well():
    try:
        from msmbuilder.example_datasets import DoubleWell
        doublewell = DoubleWell().get()
    except:
        warnings.warn('Double well dataset not found. Did you install msmb_data?')


def test_quad_well():
    try:
        from msmbuilder.example_datasets import QuadWell
        quadwell = QuadWell().get()
    except:
        warnings.warn('Quad well dataset not found. Did you install msmb_data?')


def test_metenkephalin():
    try:
        from msmbuilder.example_datasets import MetEnkephalin
        metenkephalin = MetEnkephalin().get()
    except:
        warnings.warn('Met-enkephalin dataset not found. Did you install msmb_data?')


def test_muller():
    try:
        from msmbuilder.example_datasets import MullerPotential
        muller = MullerPotential().get()
    except:
        warnings.warn('Muller potential dataset not found. Did you install msmb_data?')
