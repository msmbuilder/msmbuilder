#include "logsumexp.h"
#include "stdlib.h"

void do_forward(const float* __restrict__ log_transmat_T, const float* __restrict__ log_startprob,
             const float* __restrict__ frame_logprob, const size_t n_trajs, const size_t* __restrict__ n_observations,
             const size_t n_states, float* __restrict__ fwdlattice)
{
    int s, t, i, j;
    float work_buffer[n_states];
    float* __restrict__ _fwdlattice = fwdlattice;
    const float* __restrict__ _frame_logprob = frame_logprob;

    for (s = 0; s < n_trajs; s++) {
        for (j = 0; j < n_states; j++)
            _fwdlattice[0*n_states + j] = log_startprob[j] + _frame_logprob[0*n_states + j];
        
        for (t = 1; t < n_observations[s]; t++) {
            for (j = 0; j < n_states; j++) {
                for (i = 0; i < n_states; i++)
                    work_buffer[i] = _fwdlattice[(t-1)*n_states + i] + log_transmat_T[j*n_states + i];
          
                _fwdlattice[t*n_states + j] = logsumexp(work_buffer, n_states) + _frame_logprob[t*n_states + j];
            }
        }
        _fwdlattice += n_observations[s]*n_states;
        _frame_logprob += n_observations[s]*n_states;
    }
}
