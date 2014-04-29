#ifndef MIXTAPE_CLBAS_H
#define MIXTAPE_CLBAS_H
#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------- BLAS ------------------------------------ */


// Single precision matrix multipy
int sgemm_(const char *transa, const char *transb, const int *m,
           const int *n, const int *k, const float *alpha,
           const float *a, const int *lda, const float *b,
           const int *ldb, const float *beta, float *c, const int *ldc);


/* ---------------------------- LAPACK ------------------------------------ */

/* Computes the Cholesky factorization of a symmetric (Hermitian) 
   positive-definite matrix. */
int spotrf_(const char *uplo, const int *n, float *a, const int *lda,
            int *info);

/* Solves a system of linear equations with a triangular matrix, with multiple
   right-hand sides. */
int strtrs_(const char *uplo, const char *trans, const char *diag, const int *n, 
            const int *nrhs, const float *a, const int *lda, float *b,
            const int *ldb, int * info);

#ifdef __cplusplus
}
#endif
#endif
