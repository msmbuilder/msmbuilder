/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "LogSumExp.hpp"
#include "stdlib.h"
namespace Mixtape {

template <typename REAL>
void forward(
const double* __restrict__ log_transmat_T,
const double* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int sequence_length,
const int n_states,
double* __restrict__ fwdlattice)
{
    int t, i, j;
    REAL work_buffer[n_states];

    for (j = 0; j < n_states; j++)
        fwdlattice[0*n_states + j] = log_startprob[j] + frame_logprob[0*n_states + j];
        
    for (t = 1; t < sequence_length; t++) {
        for (j = 0; j < n_states; j++) {
            for (i = 0; i < n_states; i++)
                work_buffer[i] = fwdlattice[(t-1)*n_states + i] + log_transmat_T[j*n_states + i];
            
            fwdlattice[t*n_states + j] = logsumexp(work_buffer, n_states) + frame_logprob[t*n_states + j];
        }
    }
}

} // namespace
