import numpy as np
from sklearn.base import BaseEstimator
from sklearn import cluster

class MultiSequenceClusterMixin(object):
    def fit(self, sequences, y=None):
        concat = np.concatenate(sequences)
        lengths = [len(s) for s in sequences]
        super(MultiSequenceClusterMixin, self).fit(concat)
        self.labels_ = [self.labels_[cl - l: cl] for (cl, l) in zip(np.cumsum(lengths), lengths)]
        return self
    
    def transform(self, sequences):
        transformed = []
        for sequence in sequences:
            transformed.append(super(MultiSequenceClusterMixin, self).transform(sequence))
        return transformed
    
    def fit_transform(self, sequences):
        return self.fit(sequences).transform(sequences)


class KMeans(MultiSequenceClusterMixin, cluster.KMeans):
    pass

class MiniBatchKMeans(MultiSequenceClusterMixin, cluster.MiniBatchKMeans):
    pass

class AffinityPropagation(MultiSequenceClusterMixin, cluster.AffinityPropagation):
    pass

class DBSCAN(MultiSequenceClusterMixin, cluster.DBSCAN):
    pass

class MeanShift(MultiSequenceClusterMixin, cluster.MeanShift):
    pass
    
class SpectralClustering(MultiSequenceClusterMixin, cluster.SpectralClustering):
    pass

class Ward(MultiSequenceClusterMixin, cluster.Ward):
    pass

