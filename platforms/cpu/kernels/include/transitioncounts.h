#ifndef MIXTAPE_CPU_TRANSITIONCOUNTS_H
#define MIXTAPE_CPU_TRANSITIONCOUNTS_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

void transitioncounts(const float* __restrict__ fwdlattice,
                      const float* __restrict__ bwdlattice,
                      const float* __restrict__ log_transmat,
                      const float* __restrict__ framelogprob,
                      const int n_observations,
                      const int n_states,
                      float* __restrict__ transcounts);

#ifdef __cplusplus
}
#endif
#endif
