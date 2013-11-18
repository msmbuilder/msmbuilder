#include "logsumexp.cuh"
#include "backward.cuh"
#include <stdlib.h>

__global__ void backward4(
const float* __restrict__ log_transmat,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ bwdlattice)
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
