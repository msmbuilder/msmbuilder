/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_CPU_BACKWARD_H
#define MIXTAPE_CPU_BACKWARD_H

#include "kernels/LogSumExp.hpp"
#include "stdlib.h"
#include "stdio.h"
namespace Mixtape {

template <typename REAL>
void backward(const double* __restrict__ log_transmat,
              const double* __restrict__ log_startprob,
              const float* __restrict__ frame_logprob,
              const int sequence_length,
              const int n_states,
              REAL* __restrict__ bwdlattice)
{
    int t, i, j;
    REAL work_buffer[n_states];

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

} // namespace

#endif
