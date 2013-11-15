#include <vector>
#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "utils.h"
}
#include "CPUGaussianHMM.hpp"
using namespace Mixtape;
int main() {
    int s, t, i, j;
    const size_t n_states = 4;
    const float startProb[] = {0.25, 0.25, 0.25, 0.25};
    const size_t n_trajs = 1;
    const size_t n_observations[n_trajs] = {5};
    const int n_features = 2;
    
    float* means = (float*) malloc(n_states*n_features*sizeof(float));
    float* variances = (float*) malloc(n_states*n_features*sizeof(float));
    float* transmat = (float*) malloc(n_states*n_states*sizeof(float));
    for (i = 0; i < n_states*n_features; i++) {
        means[i] = i;
        variances[i] = (1+i);
    }
    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) 
            transmat[i*n_states + j] = 0.1 / (n_states - 1.0);
        transmat[i*n_states + i] = 0.9;
    }

    float* trajectory = (float*) malloc(n_observations[0]*n_features*sizeof(float));
    for (t = 0; t < n_observations[0]; t++)
        for (i = 0; i < n_features; i++)
            trajectory[t*n_features + i] = t*n_features + i;

    CPUGaussianHMM hmm(trajectory, n_trajs, &n_observations[0],
                       n_states, n_features);
    hmm.setMeans(means);
    hmm.setVariances(variances);
    hmm.setTransmat(transmat);
    hmm.setStartProb(startProb);

    //printf("means\n"); pprintarray(means, n_states, n_features);
    //printf("variances\n"); pprintarray(variances, n_states, n_features);
    //printf("startprob\n"); pprintarray(startProb, 1, n_states);
    //printf("transmat\n"); pprintarray(transmat, n_states, n_states);

    printf("Log Likelihood = %f\n", hmm.doEStep());
    hmm.printFwdLattice();
    hmm.printFrameLogprob();
    hmm.printBwdLattice();
    hmm.printPosteriors();
}
