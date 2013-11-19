#include "logsumexp.cu"
#include <stdlib.h>

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
