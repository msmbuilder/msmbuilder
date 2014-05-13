import numpy as np
import simtk.openmm as mm
import warnings
from mixtape.mslds import MetastableSwitchingLDS


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
            ax = pp
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)

