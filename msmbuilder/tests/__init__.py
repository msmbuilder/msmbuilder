import warnings

# Show warnings for our package
warnings.filterwarnings('always', module='msmbuilder.*')

# Show warnings for packages where we want to be conscious of warnings
warnings.filterwarnings('always', module='mdtraj.*')
warnings.filterwarnings('default', module='scipy.*')
warnings.filterwarnings('default', module='sklearn.*')
