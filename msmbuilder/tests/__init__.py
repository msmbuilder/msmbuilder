import warnings

# Show warnings for our package
warnings.filterwarnings('always', module='msmbuilder.*')

# Show warnings for packages where we want to be conscious of warnings
warnings.filterwarnings('always', module='mdtraj.*')
warnings.filterwarnings('default', module='scipy.*')
warnings.filterwarnings('default', module='sklearn.*')

# Until msmb_data is done
from msmbuilder.example_datasets import AlanineDipeptide, FsPeptide, DoubleWell, QuadWell, MullerPotential, MetEnkephalin
AlanineDipeptide().cache()
FsPeptide().cache()
DoubleWell(random_state=0).cache()
QuadWell(random_state=0).cache()
MullerPotential(random_state=0).cache()
MetEnkephalin().cache()
