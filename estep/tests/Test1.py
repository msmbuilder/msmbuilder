from __future__ import print_function
import numpy as np
from sklearn.hmm import GaussianHMM


def printArray(name, a):
    if a.ndim == 2:
        print('FloatArray2D %s(boost::extents[%d][%d]);' % ((name, ) + a.shape))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                print('%s[%d][%d] = %f;  ' % (name, i, j, a[i][j]), end='')
            print()
    elif a.ndim == 1:
        print('FloatArray1D %s(boost::extents[%d]);' % ((name, ) + a.shape))
        for j in range(a.shape[0]):
            print('%s[%d] = %f;  ' % (name, j, a[j]), end='')
    else:
        raise RuntimeError()
    print()
            
class MyGaussianHMM(GaussianHMM):
    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MyGaussianHMM, self)._accumulate_sufficient_statistics(
            stats, seq, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        printArray('refFrameLogProb', framelogprob)
        printArray('refForward', fwdlattice)
        printArray('refBackward', bwdlattice)
        printArray('refPosteriors', posteriors)
        printArray('refTrans', stats['trans'])
        printArray('refPosts', stats['post'])
        printArray('refObs', stats['obs'])
        printArray('refObs2', stats['obs**2'])
        printArray('refStart', stats['start'])
        print()
        exit(1)

hmm = MyGaussianHMM(n_components=2, init_params='', covariance_type='diag')
hmm.transmat_ = np.array([[0.7, 0.3],
                          [0.4, 0.6]])
hmm.startprob_ = np.array([0.6, 0.4])
hmm.means_ = np.array([[0], [2]])
hmm.covars_ = np.ones((2,1))
sequence = np.sin(np.arange(10)).reshape(10,1) + 1
hmm.fit([sequence])
