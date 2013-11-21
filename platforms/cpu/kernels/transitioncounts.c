#include "math.h"
#include "float.h"
#include "transitioncounts.h"
#include "logsumexp.h"

void transitioncounts(const float* __restrict__ fwdlattice,
                      const float* __restrict__ bwdlattice,
                      const float* __restrict__ log_transmat,
                      const float* __restrict__ framelogprob,
                      const int n_observations,
                      const int n_states,
                      float* __restrict__ transcounts)
{
    int i, j, t, lid;
    const int N_OBSERVATIONS32 = ((n_observations + 32 - 1) / 32) * 32;
    float work_buffer[32];
    float logprob;
    float r;
    for (i = 0; i < 32; i++)
        work_buffer[i] = -FLT_MAX;

    logprob = logsumexp(fwdlattice+(n_observations-1)*n_states, n_states);

    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) {
            r = -FLT_MAX;
            for (t = 0; t < N_OBSERVATIONS32; t++) {
                lid = t % 32;
                if (t < n_observations - 1)
                    work_buffer[lid] = fwdlattice[t*n_states+ i] + log_transmat[t*n_states + j]
                                       + framelogprob[(t + 1)*n_states + j] + bwdlattice[(t + 1)*n_states + j] - logprob;
                if (lid == 0)
                    r = logsumexp2(r, logsumexp(work_buffer, 32));
            }
            transcounts[i*n_states+j] = exp(r);
        }
    }
}
    
