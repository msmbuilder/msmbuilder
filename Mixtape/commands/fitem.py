import numpy as np
import mdtraj as md
from mixtape.cmdline import Command, argument, argument_group


class FitEM(Command):
    name = 'fit-em'
    description = 'description sdsdf sffsd'

    group_mdtraj = argument_group('MDTraj Options')
    group_mdtraj.add_argument('--dir', type=str, help='''Directory containing
        the trajectories to load''', required=True)
    group_mdtraj.add_argument('--top', type=str, help='''Topology file for
        loading trajectories''')
    group_mdtraj.add_argument('--ext', help='File extension of the trajectories',
        required=True, choices=[e[1:] for e in md.trajectory._FormatRegistry.loaders.keys()])

    group_munge = argument_group('Munging Options')
    group_vector = group_munge.add_mutually_exclusive_group(required=True)
    group_vector.add_argument('-d', '--distance-pairs', type=str,
        help='''Vectorize the MD trajectories by extracting timeseries of the
        distance between pairs of atoms in each frame. Supply a text file where
        each row contains the space-separate indices of two atoms which form a
        pair to monitor''')
    group_vector.add_argument('-a', '--atom-indices', type=str,
        help='''Superpose each MD conformation on the coordinates in the
        topology file, and then use the distance from each atom in the
        reference conformation to the corresponding atom in each MD
        conformation.''')
    group_munge.add_argument('-sp', '--split', type=int, help='''Split
        trajectories into smaller chunks. This looses some counts (i.e. like
        1%% of the counts are lost with --split 100), but can help with speed
        (on gpu + multicore cpu) and numerical instabilities that come when
        trajectories get extremely long.''', default=10000)

    group_hmm = argument_group('HMM Options')
    group_hmm.add_argument('-k', '--n-states', type=int, default=[2],
        help='Number of states in the models. Default = [2,]', nargs='+')
    group_hmm.add_argument('-l', '--lag-times', type=int, default=[1],
        help='Lag time(s) of the model(s). Default = [1,]', nargs='+')
    group_hmm.add_argument('--platform', choices=['cuda', 'cpu', 'sklearn'],
        default='cpu', help='Implementation platform. default="cpu"')
    group_hmm.add_argument('--fusion-prior', type=float, default=1e-2,
        help='Strength of the adaptive fusion prior. default=1e-2')
    group_hmm.add_argument('--n-em-iter', type=int, default=100,
        help='Maximum number of iterations of EM. default=100')
    group_hmm.add_argument('--thresh', type=float, default=1e-2,
        help='''Convergence criterion for EM. Quit when the log likelihood
        decreases by less than this threshold. default=1e-2''')
    group_hmm.add_argument('--n-lqa-iter', type=int, default=10,
        help='''Max number of iterations for local quadradric approximation
        solving the fusion-L1. default=10''')
    group_hmm.add_argument('--reversible-type', choices=['mle', 'transpose'],
        default='mle', help='''Method by which the model is constrained to be
        reversible. default="mle"''')

    group_cv = argument_group('Cross Validation')
    group_cv.add_argument('--n-cv', type=int, default=1,
        help='Run N-fold cross validation. default=1')
    # We're training and testing at the same lag time for the moment
    # group_cv.add_argument('--test-lag-time', type=int, default=1,
    #     help='Lag time at which to test the models. default=1')

    group_out = argument_group('Output')
    group_out.add_argument('-o', '--out', default='hmms.jsonlines',
        help='Output file. default="hmms.jsonlines"')


    def __init__(self, args):
        self.args = args

    def start(self):
        print 'start!'
        print self.args