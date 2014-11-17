#ifndef MIXTAPE_CLUSTER_INERTIA_H
#define MIXTAPE_CLUSTER_INERTIA_H
#ifdef __cplusplus
extern "C" {
#endif

double inertia(const double* X, const int* labels, const int* cluster_ids,
               int n_samples, int n_features, const char* metric);

#ifdef __cplusplus
}
#endif
#endif
                    