#include "math.h"
#include "string.h"
#include "cephes.h"
#include "stdio.h"
#include "stdlib.h"
#include "gammautils.h"
#define DEBUG

#if defined(_MSC_VER)
#define _ALIGNED(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define _ALIGNED(x) __attribute__ ((aligned(x)))
#endif
#endif
typedef _ALIGNED(16) float aligned_float;


int gamma_mixture(const float* restrict X, const int n_samples, const int n_features,
                  const int n_components, int n_iters, double* restrict alpha,
                  double* restrict rate, double* restrict pi)
{
    /* Fit a mixture model of gamma distributions

    Parameters
    ----------
    X : [input] array, shape=(n_samples, n_features)
        The data to model.
    alpha : [input, output] array, shape=(n_components, n_features)
        Gamma scale parameters for each components in each feature. Supply the
        starting parameters for EM here, and the final results will also be
        placed here.
    rate : [input, output] array, shape=(n_components, n_features)
        Gamma rate parameters for each components in each feature. Supply the
        starting parameters for EM here, and the final results will also be
        placed here.
    pi : [input, output] array, shape=(n_components)
        Mixture model mixing parameter. Supply the tarting parameters for EM
        here, and the final results will also be placed here.
    n_iters : int
        The number of iterations of E-M to use.
    */
    int i, j, jj, k, n;
    int err_1, err_2, err_3, err_4, err_5, err_6, err_7;
    double sum_logg_components, logsumexp_logg, p;
    double alpha_argument, new_alpha, new_pi;

    aligned_float * restrict log_X;
    double* restrict logg;
    double* restrict normalization;
    double* restrict Sum_x_p;
    double* restrict Sum_p;
    double* restrict Sum_logx_p;
    double* restrict Sum_lograte_p;
    
    // little macro for accessing the j,k-th element of an arryay that's laid
    // out with n_components x n_features, where j is the index for the
    // component and k is the index for the feature
#define JK__(J, K) J*n_features + K

    // Stuff for avoiding Nans. When pi < PI_THRESHOLD, we randomly reseed it
    // based on an adjacent component.
    static const double PI_THRESHOLD = 1e-8;
    static const double ALPHA_THRESHOLD = 1e-8;
    static const double ALPHA_OFFSET = 1.0;
    static const double RATE_OFFSET = 1.0;
    static const double PI_OFFSET = 0.001;
    double pi_renormalizer;

    err_1 = posix_memalign((void*) &log_X, 16, n_samples*n_features*sizeof(aligned_float));
    err_3 = posix_memalign((void*) &logg, 16, n_components*sizeof(double));
    err_2 = posix_memalign((void*) &normalization, 16, n_components*n_features*sizeof(double));
    err_4 = posix_memalign((void*) &Sum_p,  16, n_components*sizeof(double));
    err_5 = posix_memalign((void*) &Sum_x_p, 16, n_components*n_features*sizeof(double));
    err_6 = posix_memalign((void*) &Sum_logx_p, 16, n_components*n_features*sizeof(double));
    err_7 = posix_memalign((void*) &Sum_lograte_p, 16, n_components*n_features*sizeof(double));


    if (err_1 != 0 || err_2 != 0 || err_3 != 0 || err_4 != 0 || err_6 != 0 || err_7 != 0) {
        fprintf(stderr, "Memory Error\n");
        exit(1);
    }
    // precompute the log of the data
    for (i = 0; i < n_samples*n_features; i++)
        log_X[i] = log(X[i]);

    for (n = 0; n < n_iters; n++) {
        // precompute the log normaization factor for the log likelihood, since
        // it doesn't depend on i
        for (j = 0; j < n_components; j++)
            for (k = 0; k < n_features; k++)
                normalization[JK__(j, k)] = alpha[JK__(j, k)]*log(rate[JK__(j, k)]) - log(gamma(alpha[JK__(j, k)]));

        // clear the accumulators that sum over the samples
        memset(Sum_p, 0, n_components*sizeof(double));
        memset(Sum_x_p, 0, n_components*n_features*sizeof(double));
        memset(Sum_logx_p, 0, n_components*n_features*sizeof(double));
        memset(Sum_lograte_p, 0, n_components*n_features*sizeof(double));

        for (i = 0; i < n_samples; i++) {
            // calculate logg[j], the log likelihood that sample i is
            // in component j
            for (j = 0; j < n_components; j++) {
                logg[j] = 0.0;
                for (k = 0; k < n_features; k++) {
                    logg[j] += normalization[JK__(j, k)] \
                               + log_X[i*n_features + k]*(alpha[JK__(j, k)]-1.0f) \
                               - X[i*n_features + k]*rate[JK__(j, k)];
                }
            }

            logsumexp_logg = weightlogsumexp(logg, pi, n_components);
            for (j = 0; j < n_components; j++) {
                // this is the probability that data point i is in component j
                p = exp(log(pi[j]) + logg[j] - logsumexp_logg);

                // increment the accumulators
                Sum_p[j] += p;
                for (k = 0; k < n_features; k++) {
                    Sum_x_p[j*n_features+k] += X[i*n_features + k] * p;
                    Sum_logx_p[j*n_features+k] += log_X[i*n_features + k] * p;
                    Sum_lograte_p[j*n_features+k] += log(rate[JK__(j, k)]) * p;
                }
            }
        }

        // update rate and alpha
        for (j = 0; j < n_components; j++) {
            for (k = 0; k < n_features; k++) {
                // rate update comes first, since it depends on the old alpha
                rate[j*n_features + k] = alpha[JK__(j, k)] * Sum_p[j] / Sum_x_p[JK__(j, k)];
                // these are the A and B terms in the python code
                alpha_argument = (Sum_lograte_p[JK__(j, k)] + Sum_logx_p[JK__(j, k)]) / Sum_p[j];
                // when the fit is bad (early iterations), this conditional
                // maximum likelihood update step is not guarenteed to keep
                // alpha positive, which causes the next iteration to be f*cked.
                new_alpha = invpsi(alpha_argument);
                alpha[j*n_features + k] = new_alpha > ALPHA_THRESHOLD ? new_alpha : ALPHA_THRESHOLD;
            }
        }

        // update pi
        for (j = 0; j < n_components; j++) {
            pi[j] = Sum_p[j] / n_samples;
            if (pi[j] < PI_THRESHOLD) {
                printf("(Iteration %d) equilibrium population of state=%d is %g. Reseeding...\n", n, j, pi[j]);
                // the equilibrium popuation of this state is so low, that it
                // can cause numerical problems. Let's just 'reseed' it
                jj = (j + 1) % n_components;
                for (k = 0; k < n_features; k++) {
                    alpha[JK__(j, k)] = alpha[JK__(jj, k)] + ALPHA_OFFSET;
                    rate[JK__(j, k)] = rate[JK__(jj, k)] + RATE_OFFSET;
                }
                // put some population here so that it doesn't get ignored
                // in the next iteration. Because once a state is given pi ~ 0,
                // it is very hard to "recover".
                pi[j] = PI_OFFSET/n_components;
            }
        }

        // renormalize the PIs to add up to 1.0. This can get off
        // if one of them was less than PI_THRESHOLD, or just based on sum
        // error in the Sum_p accumulation
        pi_renormalizer = 0.0;
        for (j = 0; j < n_components; j++)
            pi_renormalizer += pi[j];
        for (j = 0; j < n_components; j++)
            pi[j] = pi[j] / pi_renormalizer;

#ifdef DEBUG
        printf("Alpha\n");
        for (j = 0; j < n_components; j++) {
            for (k = 0; k < n_features; k++)
                printf("%8g   ", alpha[j*n_features + k]);
            printf("\n");
        }

        printf("Rate\n");
        for (j = 0; j < n_components; j++) {
            for (k = 0; k < n_features; k++)
                printf("%8g   ", rate[j*n_features + k]);
            printf("\n");
        }

        printf("Pi\n[");
        for (j = 0; j < n_components; j++)
            printf("%g  ", pi[j]);
        printf("]\n\n");
#endif
    }



    free(log_X);
    free(normalization);
    free(logg);
    free(Sum_p);
    free(Sum_x_p);
    free(Sum_logx_p);
    free(Sum_lograte_p);

    return 1;
}