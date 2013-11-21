#ifndef MIXTAPE_CPU_LOGSUMEXP_H
#define MIXTAPE_CPU_LOGSUMEXP_H
#include <emmintrin.h>
#ifdef __cplusplus
extern "C" {
#endif

float logsumexp2(float v1, float v2);
float logsumexp(const float* __restrict__ buf, int N);
float _mm_logsumexp(__m128* buf, int N);

#ifdef __cplusplus
}
#endif
#endif
