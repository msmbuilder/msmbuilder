import numpy as np
import numpy as np
from sklearn.hmm import GaussianHMM
n_states = 4
n_observations = 5
n_features = 2
means = 1.0 * np.arange(n_states*n_features).reshape(n_states, n_features)
variances = 1.0 + np.arange(n_states*n_features).reshape(n_states, n_features)
trajectory = np.arange(n_observations * n_features).reshape(n_observations, n_features)

transmat = 0.1/(n_states-1.0) * np.ones((n_states, n_states))
np.fill_diagonal(transmat, 0.9)
startprob = np.ones(n_states, dtype=np.float) / n_states

hmm = GaussianHMM(n_components=n_states, init_params='', params='')
hmm.means_ = means
hmm.covars_ = variances
hmm.startprob_ = startprob
hmm.transmat_ = transmat

#print 'means\n', means
#print 'vars\n', variances
#print 'startprob\n', startprob
#print 'transmat\n', transmat

framelogprob = hmm._compute_log_likelihood(trajectory)
from sklearn.utils.extmath import logsumexp
print 'framelogprob\n', framelogprob
print 'forward\n', hmm._do_forward_pass(framelogprob)
print 'backward\n', hmm._do_backward_pass(framelogprob)

lpr, fwdlattice = hmm._do_forward_pass(framelogprob)
bwdlattice = hmm._do_backward_pass(framelogprob)
gamma = fwdlattice + bwdlattice
posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
print 'posteriors\n', posteriors
print 'logsumexp(fwdlattice[-1])', logsumexp(fwdlattice[-1])
print 'lpr', lpr
