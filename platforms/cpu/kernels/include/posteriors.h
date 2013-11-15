#ifndef MIXTAPE_POSTERIORS_H
#define MIXTAPE_POSTERIORS_H
#include <stdlib.h>
void do_posteriors(const float* __restrict__ fwdlattice, const float* __restrict__ bwdlattice,
                   const size_t n_trajs, const size_t* n_observations, const size_t n_states,
                   float* __restrict__ posteriors);
#endif
