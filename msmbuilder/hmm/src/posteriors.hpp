/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/
#ifndef MIXTAPE_CPU_POSTERIORS_H
#define MIXTAPE_CPU_POSTERIORS_H
#include "math.h"
#include "logsumexp.hpp"
namespace Mixtape {

template<typename REAL>
void compute_posteriors(const REAL* __restrict fwdlattice,
                        const REAL* __restrict bwdlattice,
                        const int sequence_length,
                        const int n_states,
                        float* __restrict posteriors)
{
    int t, i;
    REAL* gamma = (REAL*) malloc(n_states * sizeof(REAL));

    REAL normalizer;

    for (t = 0; t < sequence_length; t++) {
        for (i = 0; i < n_states; i++) {
            gamma[i] = fwdlattice[t*n_states + i] + bwdlattice[t*n_states + i];
        }
        normalizer = logsumexp(gamma, n_states);
        for (i = 0; i < n_states; i++) {
            posteriors[t*n_states+i] = exp(gamma[i] - normalizer);
        }
    }
    
    free(gamma);
}

}
#endif
