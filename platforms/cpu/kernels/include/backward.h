#ifndef MIXTAPE_CPU_BACKWARD_H
#define MIXTAPE_CPU_BACKWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

// Note that here, we use the log transition matrix, whereas
// in the forward kernel we use its transpose
void do_backward(const float* __restrict__ log_transmat, const float* __restrict__ log_startprob,
                  const float* __restrict__ frame_logprob, const size_t n_trajs, const size_t* __restrict__ n_observations,
              const size_t n_states, float* __restrict__ bwdlattice);
#ifdef __cplusplus
}
#endif
#endif
