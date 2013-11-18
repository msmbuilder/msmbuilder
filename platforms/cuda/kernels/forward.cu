#include "logsumexp.cuh"
#include "forward.cuh"
#include <stdlib.h>

__global__ void forward4(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
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


__global__ void forward8(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
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


__global__ void forward16(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
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

__global__ void forward32(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ fwdlattice)
{
    // WARPS_PER_TRAJ is the number of warps to allocate per trajectory.
    // The warps need to sync with each other at each step in the trajectory,
    // WARPS_PER_TRAJ needs to be small enough for all of the threads to fit
    // in a single block.
    const int n_states = 32;
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
