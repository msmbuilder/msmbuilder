/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "logsumexp.h"
#include "backward.h"
#include "stdlib.h"
#include "stdio.h"

void backward(const float* __restrict__ log_transmat,
              const float* __restrict__ log_startprob,
              const float* __restrict__ frame_logprob,
              const int sequence_length,
              const int n_states,
              float* __restrict__ bwdlattice)
{
    int t, i, j;
    float work_buffer[n_states];

    for (j = 0; j < n_states; j++)
        bwdlattice[(sequence_length-1)*n_states + j] = 0.0f;

    for (t = sequence_length-2; t >= 0; t--) {
        for (i = 0; i < n_states; i++) {
            for (j = 0; j < n_states; j++)
                work_buffer[j] = frame_logprob[(t+1)*n_states + j] + bwdlattice[(t+1)*n_states + j] + log_transmat[i*n_states + j];
            bwdlattice[t*n_states + i] = logsumexp(work_buffer, n_states);
        }
    }
}
