import mdtraj as md
from mixtape.cmdline import argument_group

class MDTrajInputMixin(object):
    """Mixin for a command to accept trajectory input files"""
    group_mdtraj = argument_group('MDTraj Options')
    group_mdtraj.add_argument('--dir', type=str, help='''Directory containing
        the trajectories to load''', required=True)
    group_mdtraj.add_argument('--top', type=str, help='''Topology file for
        loading trajectories''', required=True)
    group_mdtraj.add_argument('--ext', help='File extension of the trajectories',
        required=True, choices=[e[1:] for e in md.trajectory._FormatRegistry.loaders.keys()])


class GaussianFeaturizationMixin(object):
    group_munge = argument_group('Munging Options')
    group_vector = group_munge.add_mutually_exclusive_group(required=True)
    group_vector.add_argument('-d', '--distance-pairs', type=str, help='''Vectorize
        the MD trajectories by extracting timeseries of the distance
        between pairs of atoms in each frame. Supply a text file where
        each row contains the space-separate indices of two atoms which
        form a pair to monitor''')
    group_vector.add_argument('-a', '--atom-indices', type=str, help='''Superpose
        each MD conformation on the coordinates in the topology file, and then use
        the distance from each atom in the reference conformation to the
        corresponding atom in each MD conformation.''')
