#include "logsumexp.h"
#include "backward.h"
#include "stdlib.h"

void do_backward(const float* __restrict__ log_transmat, const float* __restrict__ log_startprob,
                  const float* __restrict__ frame_logprob, const size_t n_trajs, const size_t* __restrict__ n_observations,
              const size_t n_states, float* __restrict__ bwdlattice)
{
    int s, t, i, j;
    float w;
    float work_buffer[n_states];
    float* __restrict__ _bwdlattice = bwdlattice;
    const float* __restrict__ _frame_logprob = frame_logprob;

    for (s = 0; s < n_trajs; s++) {
        for (j = 0; j < n_states; j++)
            _bwdlattice[(n_observations[s]-1)*n_states + j] = 0.0f;
        
        for (t = n_observations[s]-2; t >= 0; t--) {
            for (j = 0; j < n_states; j++) {
                w = _frame_logprob[(t+1)*n_states + j] + _bwdlattice[(t+1)*n_states + j];
                for (i = 0; i < n_states; i++)
                    work_buffer[j] = w + log_transmat[i*n_states + j];
                _bwdlattice[t*n_states + j] = logsumexp(work_buffer, n_states);
            }
        }
        _bwdlattice += n_observations[s]*n_states;
        _frame_logprob += n_observations[s]*n_states;
    }
}
