#include "math.h"
#include "posteriors.h"
#include "logsumexp.h"

void do_posteriors(const float* __restrict__ fwdlattice, const float* __restrict__ bwdlattice,
                   const size_t n_trajs, const size_t* n_observations, const size_t n_states,
                   float* __restrict__ posteriors)
{
    int s, t, i, j;
    float gamma[n_states];
    float normalizer;
    const float* __restrict__ _fwdlattice = fwdlattice;
    const float* __restrict__ _bwdlattice = bwdlattice;
    float* __restrict__ _posteriors = posteriors;

    for (s = 0; s < n_trajs; s++) {
        for (t = 0; t < n_observations[s]; t++) {
            for (i = 0; i < n_states; i++)
                gamma[i] = _fwdlattice[t*n_states + i] + _bwdlattice[t*n_states + i];
            normalizer = logsumexp(gamma, n_states);
            for (i = 0; i < n_states; i++)
                _posteriors[t*n_states+i] = exp(gamma[i] - normalizer);
        }
        _fwdlattice += n_observations[s] * n_states;
        _bwdlattice += n_observations[s] * n_states;
        _posteriors += n_observations[s] * n_states;
    }
}

