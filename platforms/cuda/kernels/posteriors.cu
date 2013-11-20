#include "logsumexp.cu"
#include <stdlib.h>

__global__ void posteriors4(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const int n_trajs,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
float* __restrict__ posteriors)
{
    const int n_states = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer, normalizer;

    // we only need to do 4-wide reductions, so we group the threads
    // as only 4 per trajectory. Instead, we should forget the fact
    // that they are separate trajectories, since that doesn't matter
    // since there's no forward or backward propagation and then we
    // could work on the whole trajectory in parallel, with one
    // width-4 thread team per observation

    while (gid/4 < n_trajs) {
        const unsigned int hid = gid % 4;
        const unsigned int s = gid / 4;
        const float* _fwdlattice = fwdlattice + cum_sequence_lengths[s]*n_states;
        const float* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*n_states;
        float* _posteriors = posteriors + cum_sequence_lengths[s]*n_states;

        for (int t = 0; t < sequence_lengths[s]; t++) {
            work_buffer = _fwdlattice[t*n_states + hid] + _bwdlattice[t*n_states + hid];
            normalizer = logsumexp<4>(work_buffer);
            _posteriors[t*n_states + hid] = expf(work_buffer - normalizer);
        }


        gid += gridDim.x*blockDim.x;
    }
}

template<unsigned int n_states>
__global__ void posteriors(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const int n_trajs,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
float* __restrict__ posteriors)
{
    const int N = cum_sequence_lengths[n_trajs-1] + sequence_lengths[n_trajs-1];
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer, normalizer;

    while (gid < n_states*N) {
        const unsigned int t = gid / n_states;
        const unsigned int lid = gid % n_states;
        work_buffer = fwdlattice[t*n_states + lid] + bwdlattice[t*n_states + lid];
        normalizer = logsumexp<n_states>(work_buffer);
        posteriors[t*n_states + lid] = expf(work_buffer - normalizer);

        gid += gridDim.x*blockDim.x;
    }
}
