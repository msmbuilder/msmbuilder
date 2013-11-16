#ifndef MIXTAPE_CPU_LOGSUMEXP_H
#define MIXTAPE_CPU_LOGSUMEXP_H
#include <emmintrin.h>
#ifdef __cplusplus
extern "C" {
#endif
float logsumexp(float* buf, int N);
float _mm_logsumexp(__m128* buf, int N);
#ifdef __cplusplus
}
#endif
#endif
