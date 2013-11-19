#include <stdio.h>
#include "logsumexp.cu"

__global__ void transitioncounts(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ framelogprob,
const int n_observations,
const int n_states,
float* __restrict__  transcounts)
{
    unsigned int i, j, t;
    const unsigned int WARP_WIDTH = 32;
    const float SMALL = -1000.0f;
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const int N_OBSERVATIONS32 = ((n_observations + WARP_WIDTH - 1) / WARP_WIDTH) * WARP_WIDTH;
    const int N_STATES32 = ((n_states + WARP_WIDTH - 1) / WARP_WIDTH) * WARP_WIDTH;

    float logprob = SMALL;
    float tmp;

    for (i = gid % 32; i < N_STATES32; i += 32) {
        if (i < n_states)
            tmp = fwdlattice[(n_observations-1)*n_states + i];
        else
            tmp = SMALL;
        tmp = logsumexp<32>(tmp);
        logprob = logsumexp2(logprob, tmp);
    }

    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) {
            float r = SMALL;
            for (t = gid; t < N_OBSERVATIONS32; t += gridDim.x*blockDim.x) {
                if (t < n_observations - 1)
                    tmp = fwdlattice[t*n_states + i] + log_transmat[i*n_states + j] + framelogprob[(t+1)*n_states + j] + bwdlattice[(t+1)*n_states + j] - logprob;
                else
                    tmp = SMALL;
                tmp = logsumexp<32>(tmp);
                if ((gid % 32) == 0)
                    r = logsumexp2(tmp, r);
            }
            if (gid % 32 == 0)
                transcounts[i*n_states + j] = expf(r);
        }
    }
}
