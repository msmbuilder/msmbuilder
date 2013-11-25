import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
source = open('platforms/cuda/kernels/expectedtransitions.cu')
mod = SourceModule(source.read(), no_extern_c=True)


def transitioncounts(fwdlattice, bwdlattice, framelogprob, log_transmat):
    n_observations, n_components = fwdlattice.shape
    lneta = np.zeros((n_observations-1, n_components, n_components))
    from scipy.misc import logsumexp
    logprob = logsumexp(fwdlattice[n_observations-1, :])
    print 'logprob', logprob

    for t in range(n_observations - 1):
        for i in range(n_components):
            for j in range(n_components):
                lneta[t, i, j] = fwdlattice[t, i] + log_transmat[i, j] \
                    + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - logprob
                
    return np.exp(logsumexp(lneta, 0))



N1, N2 = 8, 3

def run(N_STATES=4):
    np.random.seed(42)
    fwdlattice = np.random.rand(N1+N2, N_STATES).astype(np.float32)
    bwdlattice = np.random.rand(N1+N2, N_STATES).astype(np.float32)
    framelogprob = np.random.rand(N1+N2, N_STATES).astype(np.float32)
    log_transmat = np.random.rand(N_STATES, N_STATES).astype(np.float32)
    sequence_lengths = np.array([N1, N2], dtype=np.int32)
    cum_sequence_lengths = np.array([0, N1], dtype=np.int32)
    transcounts = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    n_trajs = 1

    f = mod.get_function('transitioncounts%d' % N_STATES)
    f(cuda.In(fwdlattice), cuda.In(bwdlattice), cuda.In(log_transmat),
      cuda.In(framelogprob), cuda.In(sequence_lengths), cuda.In(cum_sequence_lengths), np.int32(n_trajs),
      cuda.InOut(transcounts), grid=(1,1), block=(256,1,1))

    print 'cuda transcounts'
    print transcounts

    t2_1 = transitioncounts(fwdlattice[:N1], bwdlattice[:N1], framelogprob[:N1], log_transmat)
    #t2_2 = transitioncounts(fwdlattice[N1:], bwdlattice[N1:], framelogprob[N1:], log_transmat)
    print 'reference'
    print t2_1
    ref = t2_1

    print 'error N_STATES=%d: %f' % (N_STATES, np.linalg.norm(transcounts-ref))


run(N_STATES=4)
#run(N_STATES=8)
#run(N_STATES=16)
#run(N_STATES=32)
