import pycuda.autoinit
from sklearn.utils.extmath import logsumexp
import pycuda.driver as drv
import numpy as np

def ref_forward(log_transmat_T, log_startprob, frame_logprob, n_states):
    fwdlattice = np.zeros_like(frame_logprob)
    work_buffer = np.zeros(n_states)
    for i in range(n_states):
        fwdlattice[0, i] = log_startprob[i] + frame_logprob[0, i]

    for t in range(1, frame_logprob.shape[0]):
        for j in range(n_states):
            for i in range(n_states):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat_T[j, i]
            fwdlattice[t, j] = logsumexp(work_buffer) + frame_logprob[t, j]
    return fwdlattice

def ref_backward(log_transmat, log_startprob, frame_logprob, n_states):
    bwdlattice = np.zeros_like(frame_logprob)
    work_buffer = np.zeros(n_states)
    n_observations, n_components = frame_logprob.shape
    for i in range(n_states):
        bwdlattice[n_observations-1, i] = 0
        
    for t in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = log_transmat[i, j] + frame_logprob[t + 1, j] \
                    + bwdlattice[t + 1, j]
            bwdlattice[t, i] = logsumexp(work_buffer)
    return bwdlattice



from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
__device__ inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

template <int N>
__device__ float logsumexp(float value) {
    float max = value;

    for(int offset = 1; offset < N; offset <<= 1)
        max = fmaxf(max, __shfl_down(max, offset));
    for(int offset = 1; offset < N; offset <<= 1)
        max = __shfl_up(max, offset);

    value = expf(value - max);

    for(int offset = 1; offset < N; offset <<= 1)
        value += __shfl_down(value, offset);

    value = logf(value) + max;
    for(int offset = 1; offset < N; offset <<= 1)
        value = __shfl_up(value, offset);

    return value;
}

template <int N>
__device__ float sum(float value) {
    for(int offset = 1; offset < N; offset <<= 1)
        value += __shfl_down(value, offset);
    return value;
}

/**********************************************************************/
// Log multivariate normal pdf
// Note: This would be faster if we passed in the log of the variances
//       because they are being unnecessarily recomputed in the inner
//       loop
/**********************************************************************/
extern "C" {
__global__ void gaussian_likelihood(
const float* __restrict__ sequences,
const float* __restrict__ means,
const float* __restrict__ variances,
const size_t n_trajs,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_states,
const size_t n_features,
float* __restrict__ loglikelihoods
)
{
   const unsigned int WARPS_PER_TRAJ=4;
   const unsigned int WARP_WIDTH = 32;
   const unsigned int FEATURE_WIDTH = ((n_features + WARP_WIDTH - 1) / WARP_WIDTH) * WARP_WIDTH;
   const float log_M_2_PI = 1.8378770664093453f;
   unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
   float temp;
  
   while (gid / (WARP_WIDTH*WARPS_PER_TRAJ) < n_trajs) {
       const unsigned int s = gid / (WARP_WIDTH*WARPS_PER_TRAJ);
       const unsigned int lid = gid % 32;

       for (int t = 0; t < n_observations[s]; t++) {
           for (int j = gid / WARP_WIDTH; j < n_states; j += WARPS_PER_TRAJ) {
               float accumulator = 0;
               for (int i = lid; i < FEATURE_WIDTH; i += WARP_WIDTH) {
                   if (i < n_features) {
                       const float mu = means[j*n_features + i];
                       const float sigma2 = variances[j*n_features + i];
                       const float x = sequences[trj_offsets[s] + t*n_features + i];
                       temp = -0.5*(log_M_2_PI + log(sigma2) + (x-mu)*(x-mu)/sigma2);
                   } else
                       temp = 0;    
                   accumulator += sum<32>(temp);
               }
               if (lid == 0) {
                   loglikelihoods[trj_offsets[s] + t*n_states + j] = accumulator;
               }
           }
       }
       gid += gridDim.x*blockDim.x;
   }
}
}  // extern C


/**********************************************************************/
// Backward algorithm when n_states == 4, with a warp size of 32
/**********************************************************************/
extern "C" {
__global__ void backward4(
const float* __restrict__ log_transmat,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int n_trajs,
const int* __restrict__ n_observations,
const int* __restrict__ trj_offsets,
float* __restrict__ bwdlattice
)
{
    const int n_states = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer;
    int t;

    while (gid/16 < n_trajs) {
        const unsigned int hid = gid % 16;
        const unsigned int s = gid / 16;
        const int n_obs = n_observations[s];
        
        if (hid < 4)
             bwdlattice[trj_offsets[s] + (n_obs-1)*n_states + hid] = 0;

        for (t = n_obs-2; t >= 0; t--) {
            work_buffer = bwdlattice[trj_offsets[s] + (t+1)*n_states + hid%4] + log_transmat[hid] \
                          + frame_logprob[trj_offsets[s] + (t+1)*n_states + hid%4];
            work_buffer = logsumexp<4>(work_buffer);
            if (hid % 4 == 0)
                bwdlattice[trj_offsets[s] + t*n_states + hid/4] = work_buffer;
        }
        gid += gridDim.x*blockDim.x;
    }
}
}


extern "C" {
__global__ void posteriors4(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const int n_trajs,
const int* __restrict__ n_observations,
const int* __restrict__ trj_offsets,
float* __restrict__ posteriors)
{
    const int n_states = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer, normalizer;
    int t;

    // we only need to do 4-wide reductions, so we group the threads
    // as only 4 per trajectory. Instead, we should forget the fact
    // that they are separate trajectories, since that doesn't matter
    // since there's no forward or backward propagation and then we
    // could work on the whole trajectory in parallel, with one
    // width-4 thread team per observation

    while (gid/4 < n_trajs) {
        const unsigned int hid = gid % 4;
        const unsigned int s = gid / 4;
        for (int t = 0; t < n_observations[s]; t++) {
            work_buffer = fwdlattice[trj_offsets[s] + t*n_states + hid] + bwdlattice[trj_offsets[s] + t*n_states + hid];
            normalizer = logsumexp<4>(work_buffer);
            posteriors[trj_offsets[s] + t*n_states + hid] = expf(work_buffer - normalizer);
        }


        gid += gridDim.x*blockDim.x;
    }
}
}  // extern "C"


/**********************************************************************/
// Forward algorithm when n_states == 4, with a warp size of 32
/**********************************************************************/
extern "C" {
__global__ void forward4(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int n_trajs,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
float* __restrict__ fwdlattice)
{
    const int n_states = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer;
    unsigned int t;

    while (gid/16 < n_trajs) {
        const unsigned int hid = gid % 16;
        const unsigned int s = gid / 16;

        if (hid < 4)
             fwdlattice[trj_offsets[s] + hid] = log_startprob[hid] + frame_logprob[trj_offsets[s] + hid];

        for (t = 1; t < n_observations[s]; t++) {
            work_buffer = fwdlattice[trj_offsets[s] + (t-1)*n_states + hid%4] + log_transmat_T[hid];
            work_buffer = logsumexp<4>(work_buffer);
            if (hid % 4 == 0)
                fwdlattice[trj_offsets[s] + t*n_states + hid/4] = work_buffer + frame_logprob[trj_offsets[s] + t*n_states + hid/4];
        }
        gid += gridDim.x*blockDim.x;
    }
}
} // extern C


/**********************************************************************/
// Forward algorithm when n_states == 8, with a warp size of 32
/**********************************************************************/
extern "C" {
__global__ void forward8(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
float* __restrict__ fwdlattice)
{
    const int n_states = 8;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer1, work_buffer2;
    unsigned int t;

    while (gid/32 < n_trajs) {
        const unsigned int lid = gid % 32;
        const unsigned int s = gid / 32;
        const int i = lid % 8;
        const int j1 = lid / 8;
        const int j2 = lid / 8 + 4;

        if (lid < 8)
            fwdlattice[trj_offsets[s] + lid] = log_startprob[lid] + frame_logprob[trj_offsets[s] + lid];

        for (t = 1; t < n_observations[s]; t++) {
            work_buffer1 = fwdlattice[trj_offsets[s] + (t-1)*n_states + i] + log_transmat_T[j1*n_states + i];
            work_buffer2 = fwdlattice[trj_offsets[s] + (t-1)*n_states + i] + log_transmat_T[j2*n_states + i];
            work_buffer1 = logsumexp<8>(work_buffer1);
            work_buffer1 = logsumexp<8>(work_buffer2);
            if (lid % 8 == 0) {
                fwdlattice[trj_offsets[s] + t*n_states + j1] = work_buffer1 + frame_logprob[trj_offsets[s] + t*n_states + j1];
                fwdlattice[trj_offsets[s] + t*n_states + j2] = work_buffer2 + frame_logprob[trj_offsets[s] + t*n_states + j2];
            }
        }
        gid += gridDim.x*blockDim.x;
    }
}
} // extern "C"

/**********************************************************************/
// Forward algorithm when n_states == 16, with a warp size of 32
/**********************************************************************/
extern "C" {
__global__ void forward16(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
float* __restrict__ fwdlattice)
{
    const int n_states = 16;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer1, work_buffer2;
    unsigned int t, j;

    while (gid/32 < n_trajs) {
        const unsigned int lid = gid % 32;
        const unsigned int s = gid / 32;
 
        if (lid < 16)
            fwdlattice[trj_offsets[s] + lid] = log_startprob[lid] + frame_logprob[trj_offsets[s] + lid];

        for (t = 1; t < n_observations[s]; t++) {
              for (j = 0; j < 8; j++) {
                  const int i = lid % 16;
                  const int j1 = j;
                  const int j2 = j + 8;
                  work_buffer1 = fwdlattice[trj_offsets[s] + (t-1)*n_states + i] + log_transmat_T[j1*n_states + i];
                  work_buffer2 = fwdlattice[trj_offsets[s] + (t-1)*n_states + i] + log_transmat_T[j2*n_states + i];
                  work_buffer1 = logsumexp<16>(work_buffer1);
                  work_buffer2 = logsumexp<16>(work_buffer2);

                  if (i % 16 == 0) {
                      fwdlattice[trj_offsets[s] + t*n_states + j1] = work_buffer1 + frame_logprob[trj_offsets[s] + t*n_states + j1];
                      fwdlattice[trj_offsets[s] + t*n_states + j2] = work_buffer2 + frame_logprob[trj_offsets[s] + t*n_states + j2];
                  }
              }
        }
        gid += gridDim.x*blockDim.x;
    }
}
} // extern "C"


/**********************************************************************/
/* Forward algorithm when n_states = 32 with a width 32 warp          */
/**********************************************************************/
extern "C" {
__global__ void forward32(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_states,
float* __restrict__ fwdlattice)
{
    // WARPS_PER_TRAJ is the number of warps to allocate per trajectory.
    // The warps need to sync with each other at each step in the trajectory,
    // WARPS_PER_TRAJ needs to be small enough for all of the threads to fit
    // in a single block.
    const unsigned int WARPS_PER_TRAJ = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer1;
    unsigned int t, j;

    while (gid/(32*WARPS_PER_TRAJ) < n_trajs) {
        const unsigned int lid = gid % 32;
        const unsigned int s = gid / (32*WARPS_PER_TRAJ);
        fwdlattice[trj_offsets[s] + lid] = log_startprob[lid] + frame_logprob[trj_offsets[s] + lid];

        for (t = 1; t < n_observations[s]; t++) {
            for (j = gid/32; j < n_states; j += WARPS_PER_TRAJ) {
                work_buffer1 = fwdlattice[trj_offsets[s] + (t-1)*n_states + lid] + log_transmat_T[j*n_states + lid];
                work_buffer1 = logsumexp<32>(work_buffer1);
                if (lid == 0)
                    fwdlattice[trj_offsets[s] + t*n_states + j] = work_buffer1 + frame_logprob[trj_offsets[s] + t*n_states + j];
            }
            __syncthreads();
        }
        gid += gridDim.x*blockDim.x;
    }
}
} // extern C
""", no_extern_c=1)


def test_forward32():
    n_states = 32
    length = 3
    log_transmat_T = np.arange(n_states**2).reshape(n_states, n_states).astype(np.float32)
    log_startprob = np.arange(n_states).astype(np.float32)
    frame_logprob = np.arange(length*n_states).reshape(length, n_states).astype(np.float32)
    fwdlattice = np.zeros_like(frame_logprob)

    n_trajs = 1
    lengths = np.array([length]).astype(np.int64)
    trj_offsets = np.array([0]).astype(np.int64)
    
    mod.get_function('forward32')(drv.In(log_transmat_T), drv.In(log_startprob), drv.In(frame_logprob),
                                  np.int64(n_trajs), drv.In(lengths), drv.In(trj_offsets), np.int64(n_states),
                                  drv.Out(fwdlattice),
                                  block=(128, 1, 1), grid=(1,1))
    ref = ref_forward(log_transmat_T, log_startprob, frame_logprob, n_states)

    print 'cuda'
    print fwdlattice
    print 'ref'
    print ref
    

def test_forward16():
    n_states = 16
    length = 3
    log_transmat_T = np.arange(n_states**2).reshape(n_states, n_states).astype(np.float32)
    log_startprob = np.arange(n_states).astype(np.float32)
    frame_logprob = np.arange(length*n_states).reshape(length, n_states).astype(np.float32)
    fwdlattice = np.zeros_like(frame_logprob)

    n_trajs = 1
    lengths = np.array([length]).astype(np.int64)
    trj_offsets = np.array([0]).astype(np.int64)
    
    mod.get_function('forward16')(drv.In(log_transmat_T), drv.In(log_startprob), drv.In(frame_logprob),
                                np.int64(n_trajs), drv.In(lengths), drv.In(trj_offsets),
                                drv.Out(fwdlattice),
                                block=(32, 1, 1), grid=(1,1))
    print 'cuda'
    print fwdlattice.astype(np.int)
    print 'ref'
    print ref_forward(log_transmat_T, log_startprob, frame_logprob, n_states).astype(np.int)
    

def test_forward8():
    n_states = 8
    length = 3
    log_transmat_T = np.arange(n_states**2).reshape(n_states, n_states).astype(np.float32)
    log_startprob = np.arange(n_states).astype(np.float32)
    frame_logprob = np.arange(length*n_states).reshape(length, n_states).astype(np.float32)
    fwdlattice = np.zeros_like(frame_logprob)

    n_trajs = 1
    lengths = np.array([length]).astype(np.int64)
    trj_offsets = np.array([0]).astype(np.int64)
    
    mod.get_function('forward8')(drv.In(log_transmat_T), drv.In(log_startprob), drv.In(frame_logprob),
                                np.int64(n_trajs), drv.In(lengths), drv.In(trj_offsets),
                                drv.Out(fwdlattice),
                                block=(32, 1, 1), grid=(1,1))
    print 'cuda'
    print fwdlattice.astype(np.int)
    print 'ref'
    print ref_forward(log_transmat_T, log_startprob, frame_logprob, n_states).astype(np.int)
    

def test_forward4():
    forward = mod.get_function("forward4")
    n_states = 4
    np.random.seed(42)
    log_transmat_T = np.random.rand(n_states, n_states).astype(np.float32)
    log_transmat_T = np.arange(16).reshape(4,4).astype(np.float32)
    
    n_observations = np.array([5, 8, 5, 5, 3])
    n_trajs = 5
    trj_offsets = np.concatenate(([0], np.cumsum(n_observations)[:-1])) * n_states
    log_startprob = np.random.rand(n_states).astype(np.float32)
    frame_logprob = np.random.rand(np.sum(n_observations), n_states).astype(np.float32)
    print trj_offsets, len(frame_logprob)*4
    fwdlattice = np.zeros_like(frame_logprob)
    
    forward(drv.In(log_transmat_T), drv.In(log_startprob), drv.In(frame_logprob),
            np.int32(n_trajs), drv.In(n_observations), drv.In(trj_offsets),
            drv.Out(fwdlattice),
            block=(32, 1, 1), grid=(1,1))
    
    ref_results = []
    for i in range(len(n_observations)-1):
        ref_results.append(ref_forward(log_transmat_T, log_startprob, frame_logprob[trj_offsets[i]/4:trj_offsets[i+1]/4], n_states))
    ref_results.append(ref_forward(log_transmat_T, log_startprob, frame_logprob[trj_offsets[-1]/4:], n_states))
    ref_results = np.concatenate(ref_results)
    print 'errors'
    print np.abs(ref_results - fwdlattice)
    print fwdlattice
    print ref_results


def test_gaussian_likelihood():
    np.random.seed(42)
    from sklearn.mixture.gmm import _log_multivariate_normal_density_diag
    n_states = 4
    n_features = 33
    n_observations = 10
    means = np.random.randn(n_states, n_features).astype(np.float32)
    variances = np.random.rand(n_states, n_features).astype(np.float32)
    s = np.random.randn(n_observations, n_features).astype(np.float32)
    ref = _log_multivariate_normal_density_diag(s, means, variances)

    loglikelihoods = np.zeros((n_observations, n_states), dtype=np.float32)
    mod.get_function('gaussian_likelihood')(drv.In(s), drv.In(means), drv.In(variances),
                                            np.int64(1), drv.In(np.array([n_observations])), drv.In(np.array(0)),
                                            np.int64(n_states), np.int64(n_features), drv.Out(loglikelihoods),
                                            block=(32,1,1), grid=(1,1))
    print 'reference'
    print ref
    print 'cuda'
    print loglikelihoods
                                    

def test_backward4():
    n_states = 4
    np.random.seed(42)
    log_transmat = np.random.rand(n_states, n_states).astype(np.float32)
    log_transmat = np.arange(16).reshape(4,4).astype(np.float32)    
    n_observations = np.array([10, 8, 5, 5, 3], dtype=np.int32)
    n_trajs = 5
    trj_offsets = (np.concatenate(([0], np.cumsum(n_observations)[:-1])) * n_states).astype(np.int32)
    log_startprob = np.random.rand(n_states).astype(np.float32)
    frame_logprob = np.random.rand(np.sum(n_observations), n_states).astype(np.float32)
    print trj_offsets, len(frame_logprob)*4
    bwdlattice = 13*np.ones_like(frame_logprob) + 1

    mod.get_function("backward4")(drv.In(log_transmat), drv.In(log_startprob), drv.In(frame_logprob),
                                  np.int64(n_trajs), drv.In(n_observations), drv.In(trj_offsets), 
                                  drv.Out(bwdlattice),
                                  block=(32, 1, 1), grid=(1,1))
    
    ref_results = []
    for i in range(len(n_observations)-1):
        ref_results.append(ref_backward(log_transmat, log_startprob, frame_logprob[trj_offsets[i]/4:trj_offsets[i+1]/4], n_states))
    ref_results.append(ref_backward(log_transmat, log_startprob, frame_logprob[trj_offsets[-1]/4:], n_states))
    ref_results = np.concatenate(ref_results)

    print 'cuda'
    print bwdlattice
    print 'ref'
    print ref_results


def test_posteriors4():
    n_trajs = 1
    n_observations = 10
    fwdlattice = np.random.rand(n_observations, 4).astype(np.float32)
    bwdlattice = np.random.rand(n_observations, 4).astype(np.float32)
    posteriors = np.zeros_like(fwdlattice)
    mod.get_function('posteriors4')(drv.In(fwdlattice), drv.In(bwdlattice), np.int32(n_trajs),
                                    drv.In(np.array([n_observations], dtype=np.int32)),
                                    drv.In(np.array([0], dtype=np.int32)),
                                    drv.Out(posteriors), block=(32,1,1), grid=(1,1))
    print 'cuda'
    print posteriors

    print 'reference'
    gamma = fwdlattice + bwdlattice
    print np.exp(gamma.T - logsumexp(gamma, axis=1)).T

if __name__ == '__main__':
    #test_forward32()
    #test_forward4()
    #test_forward8()
    #Test_forward16()
    #test_gaussian_likelihood()
    #test_backward4()
    test_posteriors4()
