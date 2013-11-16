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


from pycuda.compiler import SourceModule
mod = SourceModule("""
/*
template <int N>
__device__ void logsumexp(float v, float* r) {
    for (int i = 1; i < N; i*=2)
        v += __shfl_down(v, i);
    *r = v;
}*/

template <int N>
__device__ void logsumexp(float value, float* result) {
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lid = gid % 32;
    float max = value;
    for(int offset = 1; offset < N; offset <<= 1)
        max = fmaxf(max, __shfl_down(max, offset));

    max = __shfl(max, 0);
    value = __expf(value - max);

    for(int offset = 1; offset < N; offset <<= 1)
        value += __shfl_down(value, offset);

    if (lid % N == 0)
        *result = logf(value) + max;
}

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
const int n_states,
float* __restrict__ fwdlattice)
{
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
            logsumexp<4>(work_buffer, &work_buffer);
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
            logsumexp<8>(work_buffer1, &work_buffer1);
            logsumexp<8>(work_buffer2, &work_buffer2);
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
                  logsumexp<16>(work_buffer1, &work_buffer1); 
                  logsumexp<16>(work_buffer2, &work_buffer2); 

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
""", no_extern_c=1)

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
            np.int32(n_trajs), drv.In(n_observations), drv.In(trj_offsets), np.int32(n_states),
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




if __name__ == '__main__':
    test_forward4()
    test_forward8()
    test_forward16()
