#include "forward.h"
#include "assertions.h"
#include "stdlib.h"
#include "stdio.h"

int main() {
    int s, i, j, k;
    const size_t n_states = 7;
    const size_t n_observations[] = {2, 3, 4, 5, 6};
    const int n_trajs = sizeof(n_observations) / sizeof(n_observations[0]);
    int n_total_obs = 0;
    for (i = 0; i < n_trajs; i++)
        n_total_obs += n_observations[i];

    float* log_transmat_T = (float*) malloc(n_states * n_states * sizeof(float));
    if (log_transmat_T == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < n_states; i++)
        for (j = 0; j < n_states; j++)
            log_transmat_T[i*n_states + j] = 1.0/(i+j+1);

    float* log_startprob = (float*) malloc(n_states * sizeof(float));
    if (log_startprob == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < n_states; i++)
        log_startprob[i] = 0.12;

    
    float* frame_logprob = (float*) malloc(n_total_obs * n_states * sizeof(float));
    if (frame_logprob == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < n_trajs; i++) {
        for (j = 0; j < n_observations[i]; j++) {
            for (k = 0; k < n_states; k++) 
                frame_logprob[s + j*n_states + k] = 1.0 / (j+1)*k;
        }
        s += n_observations[i] * n_states;
    }


    float* fwdlattice = (float*) malloc(n_total_obs * n_states * sizeof(float));
    if (fwdlattice == NULL)
        exit(EXIT_FAILURE);

    do_forward(log_transmat_T, log_startprob, frame_logprob, n_trajs, &n_observations[0],
               n_states, fwdlattice);
}
