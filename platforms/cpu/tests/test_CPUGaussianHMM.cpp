#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "CPUGaussianHMM.hpp"
using namespace Mixtape;

int main() {
    int s, t, i, j;
    const size_t n_states = 4;
    const float startProb[] = {0.25, 0.25, 0.25, 0.25};
    const size_t n_observations[] = {2, 3, 4};
    const int n_trajs = sizeof(n_observations) / sizeof(n_observations[0]);
    const int n_features = 2;
    int n_total_obs = 0;
    for (i = 0; i < n_trajs; i++)
        n_total_obs += n_observations[i];
    
    float* means = (float*) malloc(n_states*n_features*sizeof(float));
    float* variances = (float*) malloc(n_states*n_features*sizeof(float));
    float* sequences = (float*) malloc(n_total_obs*n_features*sizeof(float));
    float* transmat = (float*) malloc(n_states*n_states*sizeof(float));
    float* _sequences = sequences;
    for (i = 0; i < n_states*n_features; i++) {
        means[i] = i;
        variances[i] = (1+i);
    }
    for (i = 0; i < n_states; i++) {
        for (j = 0; j < n_states; j++) 
            transmat[i*n_states + j] = 0.1 / (n_states - 1.0);
        transmat[i*n_states + i] = 0.9;
    }

    for (s = 0; s < n_trajs; s++) {
        for (t = 0; t < n_observations[s]; t++)
            for (i = 0; i < n_features; i++)
                _sequences[t*n_features + i] = t - 10*i;
        _sequences += n_observations[s]*n_features;
    }

    CPUGaussianHMM hmm(sequences, n_trajs, &n_observations[0], n_states, n_features);
    hmm.setMeans(means);
    hmm.setVariances(variances);
    hmm.setTransmat(transmat);
    hmm.setStartProb(startProb);
    printf("Log Likelihood = %f\n", hmm.doMStep());
}
