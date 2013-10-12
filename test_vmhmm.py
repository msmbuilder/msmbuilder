from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pp
import numpy as np
from vmhmm import VonMisesHMM
from sklearn.hmm import GaussianHMM

def test_1():
    vm = VonMisesHMM(n_components=5)
    gm = GaussianHMM(n_components=5)
    X1 = np.random.randn(100,2)
    yield lambda: vm.fit([X1])
    yield lambda: gm.fit([X1])
    
def test_2():
    vm = VonMisesHMM(n_components=2)
    vm.means_ = [[0, 0], [np.pi, np.pi]]
    vm.kappas_ = [[5, 5], [4, 5]]
    vm.transmat_ = [[0.9, 0.1], [0.1, 0.9]]

    gm = GaussianHMM(n_components=2, covariance_type='diag')
    gm.means_ = [[0, 0], [1, 1]]
    gm.covars_ = [[0.1, 0.1], [0.2, 0.2]]
    gm.transmat_ = [[0.9, 0.1], [0.1, 0.9]]

    vx, vs = vm.sample(500)
    gx, gs = gm.sample(500)    
    
    pp.clf()
    pp.scatter(gx[:,0], gx[:,1], c=gs)
    pp.savefig('test_2_gm.png')
    pp.clf()
    pp.scatter(vx[:,0], vx[:,1], c=vs)
    pp.savefig('test_2_vm.png')
#
#def test_3():
#    vm = VonMisesHMM(n_components=2)
#    vm.fit(np.random.randn(100).reshape(-1, 1))