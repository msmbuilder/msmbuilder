from __future__ import print_function
#New BSD License
#
#Copyright (c) 2007-2013 The scikit-learn developers.
#All rights reserved.
#
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#  a. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  b. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  c. Neither the name of the Scikit-learn Developers  nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#DAMAGE.

import time
import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets, cluster
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from nose.plugins.attrib import attr

from mixtape.cluster import KCenters, MiniBatchKMeans, AffinityPropagation, MeanShift, SpectralClustering, Ward


@attr('plots')
def test_kcenters_plot_cluster_comparison():
    # plot cluster comparison example from sklearn with the mixtape
    # subclasses (including KCenters)
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html


    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    pl.figure(figsize=(16, 9.5))
    pl.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                       hspace=.01)

    plot_num = 1
    for i_dataset, dataset in enumerate([noisy_circles, noisy_moons, blobs,
                                         no_structure]):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # Compute distances
        #distances = np.exp(-euclidean_distances(X))
        distances = euclidean_distances(X)

        # create clustering estimators
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = MiniBatchKMeans(n_clusters=2)
        ward_five = Ward(n_clusters=2, connectivity=connectivity)
        spectral = SpectralClustering(n_clusters=2,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        affinity_propagation = AffinityPropagation(damping=.9, preference=-200)
        kcenters = KCenters(n_clusters=2, random_state=np.random)


        for algorithm in [two_means, affinity_propagation, ms, spectral,
                          ward_five, kcenters]:
            # predict cluster memberships
            t0 = time.time()
            y_pred = np.concatenate(algorithm.fit_predict([X]))
            t1 = time.time()

            # plot
            pl.subplot(4, 6, plot_num)
            if i_dataset == 0:
                pl.title(str(algorithm).split('(')[0], size=18)
            pl.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                pl.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            pl.xlim(-2, 2)
            pl.ylim(-2, 2)
            pl.xticks(())
            pl.yticks(())
            pl.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                    transform=pl.gca().transAxes, size=15,
                    horizontalalignment='right')
            plot_num += 1

    pl.savefig('cluster-comparison.png')
    print('saving cluster-comparison.png')
