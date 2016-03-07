#ifndef METZNER_MCMC_STEP_H
#define METZNER_MCMC_STEP_H

#ifdef __cplusplus
extern "C" {
#endif

void
metzner_mcmc_step(const double* Z, const double* N, double* K,
                  double* Q, const double* random, double* sc, int n_states,
                  int n_steps);

#ifdef __cplusplus
}
#endif

#endif
