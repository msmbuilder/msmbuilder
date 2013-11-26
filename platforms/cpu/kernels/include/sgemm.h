#ifndef MIXTAPE_CPU_SGEMM_H
#define MIXTAPE_CPU_SGEMM_H
#ifdef __cplusplus
extern "C" {
#endif

int sgemm(const char *transa, const char *transb, const int *m,
          const int *n, const int *k, const float *alpha,
          const float *a, const int *lda,  float *b,
          const int *ldb, const float *beta, float *c, const int *ldc);

#ifdef __cplusplus
}
#endif
#endif
