/**
 * Author: Robert T. McGibbon
 * Contributors:
 * Copyright: 2014 Stanford University and the Authors
 *
 * Implementation of the reversible transition matrix sampler from [1].
 *
 * .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
 * Distribution of transition matrices and functions of transition
 * matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)
 */

#include <stdio.h>
#include <math.h>

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

/* Upper and lower bounds on the sum of the K matrix, to ensure proper *
/* proposal weights. See Eq. 17 of [1]. */
static const double K_MINUS = 0.9;
static const double K_PLUS = 1.1;

static double
acceptance_ratio(int i, int j, int n_states, double epsilon,
                 const double* Z,
                 const double* K,
                 const double* N,
                 const double* Q)
{
    double term1, term2, term3, term4;
    term1 = Z[i*n_states + j] * (log(K[i*n_states + j] + epsilon) - log(K[i*n_states + j]));
    term3 = N[i] * (log(Q[i]) - log(Q[i] + (epsilon)));
    if (i == j)
        return exp(term1 + term3);

    term2 = Z[j*n_states + i] * (log(K[j*n_states + i] + epsilon) - log(K[j*n_states + i]));
    term4 = N[j] * (log(Q[j]) - log(Q[j] + (epsilon)));
    return exp(term1 + term2 + term3 + term4);
}


/**
 * Run n_steps of Metropolis MCMC for the Metzner reversible transition
 * matrix sampler
 * 
 * Parameters
 * ----------
 * Z : [in] array, shape=(n_states, n_states)
 *     Effective number of transition counts (the sufficient statistic)
 * N : [in] array, shape=(n_states)
 *     Row-sums of Z
 * K : [in/out] array, shape=(n_states, n_states)
 *     The parameters being sampled -- the "virtual counts". These parameters
 *     are directly modified when a step is accepted. Note that K is symmetric,
 *     and will stay symmetric.
 * Q : [in/out] array, shape=(n_states)
 *     Row-sums of K. This will 
 * random : [in] array, shape=(n_steps * 4)
 *     Array of n_steps*4 random doubles in [0, 1). The quality of the C
 *     stdlib's random number generation is pretty flakey, so instead this
 *     requires the caller to pass in random numbers its own source (e.g. numpy)
 * n_states : int
 *     The dimension for the matrices
 * n_steps : int
 *     The number of MCMC steps to take. The K and Q matrices will be updated
 *     on acceptances.
 */

void
metzner_mcmc_step(const double* Z, const double* N, double* K,
                  double* Q, const double* random, double* sc, int n_states,
                  int n_steps)
{
    int i, j, t;
    double a, b, r, epsilon, cutoff;

    for (t = 0; t < n_steps; t++) {
        i = (int) ((*(random++)) * n_states);
        j = (int) ((*(random++)) * n_states);

        if (i == j) {
            a = MAX(-K[i*n_states+j], K_MINUS - (*sc));
            b = K_PLUS - (*sc);
        } else {
            a = MAX(-K[i*n_states+j], 0.5*(K_MINUS - (*sc)));
            b = 0.5 * (K_PLUS - (*sc));
        }

        epsilon = a + (*(random++)) * (b-a);
        cutoff = acceptance_ratio(i, j, n_states, epsilon, Z, K, N, Q);
        r = (*(random++));
        /* printf("i=%d, j=%d\n", i, j); */
        /* printf("a=%f, b=%f\n", a, b); */
        /* printf("epsilon=%f\n", epsilon); */
        /* printf("cutoff=%f\n", cutoff); */
        /* printf("sc = %f\n", (*sc)); */
        /* printf("r=%f\n", r); */

        if (r < cutoff) {
            K[i*n_states + j] += epsilon;
            (*sc) += epsilon;
            Q[i] += epsilon;
            if (i != j) {
                K[j*n_states + i] += epsilon;
                (*sc) += epsilon;
                Q[j] += epsilon;
            }
        }
    }
}
