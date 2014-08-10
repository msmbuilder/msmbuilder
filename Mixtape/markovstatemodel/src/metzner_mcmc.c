#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
static const double K_MINUS = 0.9;
static const double K_PLUS = 1.1;

static double
sum_array(const double* A, int n)
{
    int i;
    double sum = 0;
    for (i = 0; i < n; i++)
         sum += A[i];
    return sum;
}


static void
symmetric_indices(int idx, int n, int* i, int* j)
{
    //Given linear index k, and n the size of nxn matrix
    *i = floor((2*n+1 - sqrt((2*n+1)*(2*n+1) - 8*idx)) / 2);
    *j = idx - (n-1)*(*i) + (*i)*((*i)-1)/2;
    
    // index = j + n*i - i (i-1) / 2
}


double
acceptance_ratio(int k, int n_states, double old, double new,
                 const double* S,
                 const double* N,
                 const double* K,
                 const double* Q)
{
    int i, j, n_sym;
    double term1 = 0, term2 = 0, term3 = 0;
    symmetric_indices(k, n_states, &i, &j);
    term1 = S[k] * (log(new) - log(old));
    term2 = N[i] * (log(Q[i]) - log(Q[i] + (new - old)));
    term3 = N[j] * (log(Q[j]) - log(Q[j] + (new - old)));
    return exp(term1 + term2 + term3*(i==j?0:1));
}


void
metzner_mcmc_step(const double* K,  int n_states, int n_samples, int n_thin,
                  const double* random, double* out);

void
metzner_mcmc(const double* Z, int n_states, int n_samples, int n_thin,
             const double* random, double* out)
{

    int i, j, k, t;
    const int n_sym = n_states * (n_states + 1) / 2;
    double *S, *K, *N, *Q;
    double cutoff, epsilon, a, b, r, sc = 0, zc = 0;


    /* set S to hold the upper triangular entries of Z + Z.T */
    /* without double counting the diagonal */
    S = (double*) malloc(n_sym * sizeof(double));
    K = (double*) malloc(n_sym * sizeof(double));
    N = (double*) malloc(n_states * sizeof(double));
    Q = (double*) malloc(n_states * sizeof(double));
    memset(N, 0, n_states*sizeof(double));
    memset(Q, 0, n_states*sizeof(double));

    k = 0;
    for (i = 0; i < n_states; i++) {
        /* only add the diagonal once */
        S[k++] = Z[i*n_states + i];
        for (j = i+1; j < n_states; j++)
            S[k++] = Z[i*n_states + j] + Z[j*n_states + i];

        // set the row sums
        for (j = 0; j < n_states; j++)
            N[i] += Z[i*n_states + j];
    }
    // initialize K = 0.5 * (Z + Z.T) / sum(Z)
    k = 0;
    zc = sum_array(Z, n_states * n_states);
    for (i = 0; i < n_states; i++)
        for (j = i; j < n_states; j++)
            K[k++] = 0.5 * (Z[i*n_states + j] + Z[j*n_states + i]) / zc;

    sc = 1;

    // set Q, the row sums of K, taking care for the symmetric storage
    for (k = 0; k < n_sym; k++) {
        symmetric_indices(k, n_states, &i, &j);
        Q[i] += K[k];
        if (i != j)
            Q[j] += K[k];
    }

    printf("S = %f %f %f %f %f %f\n", S[0], S[1], S[2], S[3], S[4], S[5]);
    printf("K = %f %f %f %f %f %f\n", K[0], K[1], K[2], K[3], K[4], K[5]);
    printf("N = %f %f %f\n", N[0], N[1], N[2]);
    printf("Q = %f %f %f\n\n", Q[0], Q[1], Q[2]);


    for (t = 0; t < n_samples; t++) {
        k = (int) (*(random++)) * n_sym;

        symmetric_indices(k, n_states, &i, &j);
        if (i == j) {
            a = MAX(-K[k], K_MINUS - sc);
            b = K_PLUS - sc;
        } else {
            a = MAX(-K[k], 0.5*(K_MINUS - sc));
            b = 0.5 * (K_PLUS - sc);
        }

        epsilon = a + (*(random++)) * (b-a);
        epsilon = 0.01;
        cutoff = acceptance_ratio(k, n_states, K[k], K[k] + epsilon, S, N, K, Q);
        printf("i=%d, j=%d\n", i, j);
        printf("a=%f, b=%f\n", a, b);
        printf("epsilon=%f\n", epsilon);
        printf("cutoff=%f\n", cutoff);
        r = (*(random++));
        
        if (r < cutoff) {
            printf("Accept");
        }
    }

}




// @cython.wraparound(False)
// @cython.boundscheck(False)
// def logsymsumexp(np.ndarray[ndim=1, dtype=DTYPE_T] x not None,
//                  int n):
//     """Calculate the log-sum-exp of the rows (or columns) of a symmetric
//     matrix stored in symmetric storage.
//
//     Symmetric storage is a simple format for storing a 2d symmetric matrix
//     in a 1 dimensional array of length N*(N+1)/2. For a 5x5 matrix, the
//     indices in the 1-d vector for each upper triangular matrix entry are as
//     shown below:
//
//     [0  1  2   3  4]
//     [-  5  6   7  8]
//     [-  -  9  10 11]
//     [-  -  -  12 13]
//     [-  -  -  -  14]
//     """
//     cdef int i, j, k
//     if n*(n+1)//2 != len(x):
//         raise ValueError('Incompatible vector size. It must be a binomial '
//                          'coefficient n choose 2 for some integer n >= 2.')
//     log_sums = np.zeros(n, dtype=DTYPE)
//     maxes = np.empty(n, dtype=DTYPE)
//     maxes.fill(-HUGE_VAL)
//
//     k = 0
//     for i in range(n):
//         for j in range(i, n):
//             maxes[j] = max(x[k], maxes[j])
//             maxes[i] = max(x[k], maxes[i])
//             k += 1
//
//     k = 0
//     for i in range(n):
//         log_sums[i] += exp(x[k] - maxes[i])
//         k += 1
//         for j in range(i+1, n):
//             log_sums[i] += exp(x[k] - maxes[i])
//             log_sums[j] += exp(x[k] - maxes[j])
//             k += 1
//
//     for i in range(n):
//         log_sums[i] = log(log_sums[i]) + maxes[i]
//     return log_sums
//





/* -------- */
int main(int argc, char** argv)
{
    const double Z[3][3] = {{1, 10, 2}, {2, 26, 3}, {15, 20, 20}};
    double out[1];
    double random[2] = {0, 1};
    metzner_mcmc((double*) Z, 3, 1, 1, random, out);
}


