import numpy as np
import simtk.openmm as mm
import mdtraj as md
import os
import mdtraj.reporters
from simtk.openmm import app
from simtk import unit
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import warnings
from mixtape.mslds import MetastableSwitchingLDS
import matplotlib.pyplot as plt


class PlusminModel():
    """This example system has the following one-dimensional dynamics:
        x_{t+1}^1 = x_t + \epsilon_1
        x_{t+1}^2 = -x_t + \epsilon_2
    """

    def __init__(self):
        self.K = 2
        self.x_dim = 1
        self.As = np.reshape(np.array([[0.6], [0.6]]),
                                (self.K, self.x_dim, self.x_dim))
        self.bs = np.reshape(np.array([[0.4], [-0.4]]),
                                (self.K, self.x_dim))
        self.Qs = np.reshape(np.array([[0.01], [0.01]]),
                                (self.K, self.x_dim, self.x_dim))
        self.Z = np.reshape(np.array([[0.995, 0.005], [0.005, 0.995]]),
                                (self.K, self.K))
        self.pi = np.reshape(np.array([0.99, 0.01]), (self.K,))
        self.mus = np.reshape(np.array([[1], [-1]]), (self.K, self.x_dim))
        self.Sigmas = np.reshape(np.array([[0.01], [0.01]]),
                                (self.K, self.x_dim, self.x_dim))

        # Generate Solver
        s = MetastableSwitchingLDS(self.K, self.x_dim)
        s.As_ = self.As
        s.bs_ = self.bs
        s.Qs_ = self.Qs
        s.transmat_ = self.Z
        s.populations_ = self.pi
        s.means_ = self.mus
        s.covars_ = self.Sigmas
        self._model = s

    def generate_dataset(self, n_seq, T):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        obs_sequences = []
        hidden_sequences = []
        for n in range(n_seq):
            xs, Ss = self._model.sample(T)
            obs_sequences.append(xs)
            hidden_sequences.append(Ss)
        return obs_sequences, hidden_sequences


class MullerForce(mm.CustomExternalForce):

    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j],
                       cc=self.cc[j], AA=self.AA[j],
                       XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2
                         + {bb} * (x - {XX}) * (y - {YY})
                         + {cc} * (y - {YY})^2)'''.format(**fmt)
        super(MullerForce, self).__init__(expression)

    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            try:
                value += cls.AA[j] * np.exp(
                    cls.aa[j] * (x - cls.XX[j]) ** 2 +
                    cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) +
                    cls.cc[j] * (y - cls.YY[j]) ** 2)
            except FloatingPointError:
                value = np.exp(100)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2,
             maxy=2, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx - minx, maxy - miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx: maxx: grid_width, miny: maxy: grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)

class MullerModel():
    def __init__(self):
        # each particle is totally independent
        self.nParticles = 1
        self.mass = 1.0 * dalton
        # temps  = 200 300 500 750 1000 1250 1500 1750 2000
        self.temperature = 200 * kelvin
        self.friction = 100 / picosecond
        self.timestep = 10.0 * femtosecond

        self.x_dim = 2
        self.y_dim = 2
        self.K = 3

    def generate_dataset(self, n_seq, num_trajs, T):
        # TODO: Clean this function up
        # Choose starting conformations uniform on the grid
        # between (-1.5, -0.2) and (1.2, 2)
        start = T / 4 # Not sure if this is necessary
        xs = np.zeros((n_seq, num_trajs * (T - start), self.y_dim))
        for traj in range(n_seq):
            system = mm.System()
            mullerforce = MullerForce()
            for i in range(self.nParticles):
                system.addParticle(self.mass)
                mullerforce.addParticle(i, [])
            system.addForce(mullerforce)

            integrator = mm.LangevinIntegrator(self.temperature,
                    self.friction, self.timestep)
            context = mm.Context(system, integrator)
            startingPositions = ((np.random.rand(self.nParticles, 3)
                                    * np.array([2.7, 1.8, 1]))
                                + np.array([-1.5, -0.2, 0]))

            context.setPositions(startingPositions)
            context.setVelocitiesToTemperature(self.temperature)

            trajectory = np.zeros((T, 2))
            for i in range(T):
                x = (context.getState(getPositions=True)
                        .getPositions(asNumpy=True)
                        .value_in_unit(nanometer))
                # Save the state
                if i > start:
                    xs[0, traj * (T-start) + (i-start), :] = x[0, 0:2]
                trajectory[i, :] = x[0, 0:2]
                integrator.step(10)
        #import pdb
        #pdb.set_trace()
        return [xs[0]], trajectory, start

class AlanineDipeptideModel:
    """Generates a short Alanine Dipeptide Trajectory.
       BROKEN CURRENTLY.
    """
    def __init__(self):
        pass

    def load_dataset(self, traj_filename):
        traj = md.load(traj_filename)
        topology = traj.topology
        traj.superpose(traj[0])
        (T, N_atoms, dim) = np.shape(traj.xyz)
        y_dim = N_atoms * dim
        x_dim = y_dim
        ys = np.reshape(traj.xyz, (T, y_dim))
        ys = [ys]
        return ys, x_dim

    def generate_dataset(self, traj_filename, T):
        pdb = md.load('native.pdb')
        topology = pdb.topology.to_openmm()
        forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
        system = forcefield.createSystem(topology,
                  nonbondedMethod=app.CutoffNonPeriodic)
        integrator = mm.LangevinIntegrator(330*unit.kelvin,
          1.0/unit.picoseconds, 2.0*unit.femtoseconds)
        simulation = app.Simulation(topology, system, integrator)

        simulation.context.setPositions(pdb.xyz[0])
        simulation.context.setVelocitiesToTemperature(330*unit.kelvin)
        if os.path.exists(traj_filename):
            os.remove(traj_filename)
        simulation.reporters.append(
                md.reporters.HDF5Reporter(traj_filename, 10))
        simulation.step(T)
