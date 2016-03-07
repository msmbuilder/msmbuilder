#ifndef TRANSMAT_MLE_PRINZ_H
#define TRANSMAT_MLE_PRINZ_H

#ifdef __cplusplus
extern "C" {
#endif

int transmat_mle_prinz(const double* C, int n_states, double tol,
                       double* T, double* pi);
#ifdef __cplusplus
}
#endif

#endif
