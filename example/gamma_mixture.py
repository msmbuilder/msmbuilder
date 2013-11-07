import scipy.stats
import itertools
import numpy as np
from mixtape.gammamixture import GammaMixtureModel

data = []
n_features = 3
for i in range(n_features):
    data.append(np.concatenate((scipy.stats.distributions.gamma(1, (i+1)*20).rvs(1000),
                                scipy.stats.distributions.gamma(10, (i+1)*20).rvs(3000))))
data = np.vstack(data).T
print data.shape
data = data[np.random.permutation(len(data))]
test = data[0:len(data)/5]
train = data[len(data)/5:]

n_components = range(1,10)
bics = []
test_ll = []
for i in n_components:
    gmm = GammaMixtureModel(n_components=i, n_iter=1000)
    gmm.fit(train)
    bics.append(gmm.bic(train))
    test_ll.append(gmm.score(test).sum())
    print bics
    print test_ll

import matplotlib.pyplot as pp
pp.subplot(211)
pp.plot(n_components, bics, 'x-', c='g', label='bics')
pp.legend(loc=4)
pp.gca().twinx().plot(n_components, test_ll, 'x-', c='k', label='testll')
pp.legend(loc=1)
pp.xlabel('n states')


pp.subplot(212)
colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'k'])
for i in range(n_features):
    pp.hist(data[:, i], bins=15, color=next(colors), alpha=0.3, label='feature %d' % i)
pp.legend()
pp.show()
