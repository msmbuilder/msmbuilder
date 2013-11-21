#ifndef MIXTAPE_CPU_POSTERIORS_H
#define MIXTAPE_CPU_POSTERIORS_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif


void compute_posteriors(const float* __restrict__ fwdlattice,
                        const float* __restrict__ bwdlattice,
                        const int sequence_length,
                        const int n_states,
                        float* __restrict__ posteriors);

#ifdef __cplusplus
}
#endif
#endif
