#ifndef MIXTAPE_CPU_BACKWARD_H
#define MIXTAPE_CPU_BACKWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

void backward(const float* __restrict__ log_transmat,
              const float* __restrict__ log_startprob,
              const float* __restrict__ frame_logprob,
              const int sequence_length,
              const int n_states,
              float* __restrict__ bwdlattice);

#ifdef __cplusplus
}
#endif
#endif
