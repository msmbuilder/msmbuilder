import numpy as np
import random
from msmbuilder.tpt import mfpt
from msmbuilder.msm import MarkovStateModel
import matplotlib.pyplot as plt

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
msm = MarkovStateModel()
msm.transmat_ = tprob
msm.countsmat_ = counts
output = mfpt.mfpts(msm, sinks, errors=True, n_samples=1000)[:,1] # State 1 to 0 MFPT distribution with good statistics, approximately Gaussian/symmetric

# Make state 1 out transitions poorly sampled, with either 0 or 1 "out counts" to every other state
for i in range(10):
    counts[1][i] = random.choice([1,0])
counts[1][1] = random.choice(counts_range)
tprob = (counts.transpose() / np.sum(counts, axis=1)).transpose()
# Insert into MSM template
msm.transmat_ = tprob
msm.countsmat_ = counts
bad_output = mfpt.mfpts(msm, sinks, errors=True, n_samples=1000)[:,1] # State 1 to 0 MFPT distribution with bad state 1 out-transition statistics, exponential-like with large outliers

# Plot results
fig = plt.figure()
plot = fig.add_subplot(111)

plt.hist(output)
plt.hist(bad_output)
plt.show()
