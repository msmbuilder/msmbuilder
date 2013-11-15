#include "logsumexp.cu"
/*
 * Run the forward portion of the forward backward algorithm.
 * This kernel assumes that the number of states is equal to 32,
 * the width of the warp. Models with fewer than 32 states can be
 * padded with fake states. Models with more than 32 states are not
 * supported, because the reductions currently are only warp-wide.
 */ 
__global__ void do_forward(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ trj_offset,
const size_t* __restrict__ n_observations,
const size_t n_states,
float* __restrict__ fwdlattice)
{

    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int warpWidth = 32;
    const unsigned int lid = gid % warpWidth;
    unsigned int s, t, j;
    float work_buffer = 0;

    while(gid < n_trajs*n_states) {
        // s = the trajectory index we're working on. (up to N_TRAJS)
        s = gid / warpWidth;

        fwdlattice[trj_offset[s] + 0*n_states + lid] = log_startprob[lid] + frame_logprob[trj_offset[s] + 0*n_states + lid];

        for(t = 1; t < n_observations[s]; t++) {
            for(j = 0; j < n_states; j++) {
                work_buffer = fwdlattice[trj_offset[s] + (t-1)*n_states + lid] + log_transmat_T[j*n_states+lid];
                warplogsumexp(work_buffer, &work_buffer);
                if (lid == 0) {
                    fwdlattice[trj_offset[s] + t*n_states+j] = work_buffer + frame_logprob[trj_offset[s]+ t*n_states + j];
                }
            }
        }

        gid += gridDim.x*blockDim.x;
    }
}
