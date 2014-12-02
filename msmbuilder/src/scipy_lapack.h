#ifndef MIXTAPE_SCIPY_LAPACK
#define MIXTAPE_SCIPY_LAPACK

#include <Python.h>
#include "f2py/f2pyptr.h"

typedef int sgemm_t(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha, const float *a, const int *lda,  float *b, const int *ldb, const float *beta, float *c, const int *ldc);
typedef int spotrf_t(const char *uplo, const int *n, float *a, const int *lda, int *info);
typedef int strtrs_t(const char *uplo, const char *trans, const char *diag, const int *n, const int *nrhs, const float *a, const int *lda, float *b, const int *ldb, int * info);

typedef struct {
  sgemm_t *sgemm;
  spotrf_t *spotrf;
  strtrs_t *strtrs;
} lapack_t;
static lapack_t __lapack;


static lapack_t* get_lapack(void) {
  PyObject *mod_lapack, *mod_blas, *func, *cpointer;
  if (__lapack.sgemm == NULL) {
    mod_blas = PyImport_ImportModule("scipy.linalg.blas");
    mod_lapack = PyImport_ImportModule("scipy.linalg.lapack");

    func = PyObject_GetAttrString(mod_blas, "sgemm");
    cpointer = PyObject_GetAttrString(func, "_cpointer");
    __lapack.sgemm = (sgemm_t*) f2py_pointer(cpointer);

    func = PyObject_GetAttrString(mod_lapack, "spotrf");
    cpointer = PyObject_GetAttrString(func, "_cpointer");
    __lapack.spotrf = (spotrf_t*) f2py_pointer(cpointer);
    
    func = PyObject_GetAttrString(mod_lapack, "strtrs");
    cpointer = PyObject_GetAttrString(func, "_cpointer");
    __lapack.strtrs = (strtrs_t*) f2py_pointer(cpointer);
  }

  return &__lapack;
}

#endif
