/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/
#ifndef MIXTAPE_CPU_TRANSITIONCOUNTS_H
#define MIXTAPE_CPU_TRANSITIONCOUNTS_H
#include "math.h"
#include "float.h"
#include "stdio.h"
#include "stdlib.h"
#include "LogSumExp.hpp"
namespace Mixtape {

template <typename REAL>
void transitioncounts(const REAL* __restrict__ fwdlattice,
                      const REAL* __restrict__ bwdlattice,
                      const double* __restrict__ log_transmat,
                      const float* __restrict__ framelogprob,
                      const int n_observations,
                      const int n_states,
                      double* __restrict__ transcounts,
                      double* logprob)
{
    int i, j, t;
    REAL* work_buffer;
    work_buffer = (REAL*) malloc((n_observations-1)*sizeof(REAL));
    *logprob = logsumexp(fwdlattice+(n_observations-1)*n_states, n_states);

    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) {
            for (t = 0; t < n_observations - 1; t++) {
                work_buffer[t] = fwdlattice[t*n_states + i] + log_transmat[i*n_states + j]
                                 + framelogprob[(t + 1)*n_states + j] + bwdlattice[(t + 1)*n_states + j] - *logprob;
            }
            transcounts[i*n_states+j] = expf(logsumexp(work_buffer, n_observations-1));
        }
    }
    free(work_buffer);
}

} // namespace
#endif
