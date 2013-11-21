#include "math.h"
#include "float.h"
#include "stdio.h"
#include "stdlib.h"
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
    int i, j, t;
    float* work_buffer;
    float logprob;

    work_buffer = (float*) malloc((n_observations-1)*sizeof(float));    
    logprob = logsumexp(fwdlattice+(n_observations-1)*n_states, n_states);

    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) {
            for (t = 0; t < n_observations - 1; t++)
                work_buffer[t] = fwdlattice[t*n_states + i] + log_transmat[i*n_states + j]
                                 + framelogprob[(t + 1)*n_states + j] + bwdlattice[(t + 1)*n_states + j] - logprob;
            transcounts[i*n_states+j] = exp(logsumexp(work_buffer, n_observations-1));
        }
    }
    free(work_buffer);
}
