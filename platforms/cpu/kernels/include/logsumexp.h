#ifndef MIXTAPE_LOGSUMEXP_H
#define MIXTAPE_LOGSUMEXP_H
#include <emmintrin.h>
float logsumexp(float* buf, int N);
float _mm_logsumexp(__m128* buf, int N);
#endif
