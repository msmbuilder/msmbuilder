import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))


ax = pp.subplot(axisbg='w')
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2)
pp.plot(X_plot[:, 0], true_dens, 'k-', lw=2, label='input distribution')
        
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
log_dens = kde.score_samples(X_plot)
ax.plot(X_plot[:, 0], np.exp(log_dens), '-', lw=2, c='r', label='Gaussian KDE')
pp.twinx().hist(X, bins=20, alpha=0.5, label='Histogram')


ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')

ax.set_xlim(-4, 9)
ax.set_ylim(0, 0.4)
pp.savefig('_static/kde-vs-histogram.png')

             