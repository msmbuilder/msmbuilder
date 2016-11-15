import numpy as np
import random
from msmbuilder.tpt import mfpt
import pickle

# Create fake 10x10 counts and tprob matrices
ndim = 10
sinks = [0] # Make state 0 the sink
counts_range = range(100,200)
counts = np.zeros((ndim, ndim))
for i in range(ndim):
    for j in range(ndim):
        counts[i][j] = random.choice(counts_range)
tprob = (counts.transpose() / np.sum(counts, axis=1)).transpose()
# Insert into MSM template
msm = pickle.load(open('msm.pkl'))
msm.transmat_ = tprob
msm.countsmat_ = counts
output = mfpt.mfpts(msm, sinks, errors=True, n_samples=1000)
np.savetxt('1-0_output', output[:,1]) # MFPT distribution of state 1 to 0 with good statistics, approximately Gaussian

# Make state 1 out transitions poorly sampled, with either 0 or 1 "out counts" to every other state
for i in range(10):
    counts[1][i] = random.choice([1,0])
counts[1][1] = random.choice(counts_range)
tprob = (counts.transpose() / np.sum(counts, axis=1)).transpose()
# Insert into MSM template
msm.transmat_ = tprob
msm.countsmat_ = counts
output = mfpt_new.mfpts(msm, sinks, errors=True, n_samples=1000)
np.savetxt('1-0_undersampled_output', output[:,1]) # MFPT distribution with bad statistics, contains a few extremely large outliers
