#include "logsumexp.h"
#include "stdlib.h"
/**
 * Run the "forward" step of the forward-backward algorithm
 *
 * Parameters
 * ----------
 * log_transmat_T, array of shape [n_states, n_states]
 *     Transpose of the log of the transition probability matrix
 * log_startprob, array of shape [n_states]
 *     The log probability of the chain starting in each hidden state
 * frame_logprob, array
 *     The log probability of each frame emmitting from each state, P(X_t | S_i).
 *     The indexing is slightly complex. It's a 3D array with these probabilities
 *     for each trajectory up to `n_trajs`. If all of the trajectories are equal length,
 *     then its a simple rectangular array of shape [n_trajs, n_observations, n_states],
 *     But if they're not the same length, then its basically a concatenation of the 
 *     2D [n_observations, n_states] arrays. The total length of the array is
 *     sum(n_observations)*n_states
 * n_trajs, int
 *     Number of trajectories
 * n_observations, array of shape [n_trajs]
 *     Length of each trajectory
 * n_states, int
 *     The number of hidden states
 * fwdlattice, array of shape matching frame_logprob
 *     Output data, where the forward probability of each observation from each trajectory
 *     in each state will be stored.
 */
void do_forward(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ n_observations,
const size_t n_states,
float* __restrict__ fwdlattice)
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
