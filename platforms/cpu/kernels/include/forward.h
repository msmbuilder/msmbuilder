#ifndef MIXTAPE_CPU_FORWARD_H
#define MIXTAPE_CPU_FORWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

void forward(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int sequence_length,
const int n_states,
float* __restrict__ fwdlattice);

#ifdef __cplusplus
}
#endif
#endif
