import numpy as np
import mixtape.tica


class TICAScoreMixin(object):
    """Provides compare() and summarize() functionality for TICAOptimizer.
    Uses variational eigenvalue comparison to rank models.
    """
    def compare(self):
        self.accept = is_better(self.model.eigenvalues_, self.old_model.eigenvalues_)
        self.summarize()
        return self.accept

    def summarize(self):
        print("%d %.5f %.4f %.4f **** %.5f %.4f %.4f" % (
        self.accept, self.old_model.eigenvalues_[0], self.old_model.eigenvalues_[1], self.old_model.eigenvalues_[2], self.model.eigenvalues_[0], self.model.eigenvalues_[1], self.model.eigenvalues_[2]))


class TICAOptimizer(TICAScoreMixin):
    """Optimize TICA objective function by swapping active features one-by-one."""
    def __init__(self, subset_featurizer, trajectories):
        self.trj0 = trajectories[0][0]
        
        self.subset_featurizer = subset_featurizer
        self.build_full(trajectories)

    @property
    def subset(self):
        return self.subset_featurizer.subset
    
    def build_full(self, trajectories):
        """Featurize all active subset and build a tICA model."""
        tica = mixtape.tica.tICA()
        features = map(lambda trj: self.subset_featurizer.featurize(trj), trajectories)
        unused_output = map(lambda X: tica.partial_fit(X), features)
        
        self.model, self.features = tica, features

    def build_partial(self, trajectories, to_replace, new_value):
        """Featurize single proposal and build a tICA model."""
        tica = mixtape.tica.tICA()
        self.old_feature = map(lambda X: X[:, [to_replace]].copy(), self.features)
        new_feature = map(lambda trj: self.subset_featurizer.featurize_single(trj, new_value), trajectories)
        set_feature(self.features, new_feature, to_replace)
        unused_output = map(lambda X: tica.partial_fit(X), self.features)
        
        self.model = tica
        

    def revert(self):
        """Revert to cached features and TICA model."""
        self.model = self.old_model
        self.subset_featurizer.set(self.old_subset)
        set_feature(self.features, self.old_feature, self.to_replace)        
        
    def propose(self, trajectories):
        """Propose (and store) a single feature swap."""
        to_replace, new_value, old_value = self.subset_featurizer.propose()
        self.to_replace = to_replace
        
        self.old_model = self.model
        self.old_subset = self.subset.copy()
        
        self.subset[to_replace] = new_value
        self.subset_featurizer.set(self.subset)
        
        self.build_partial(trajectories, to_replace, new_value)
        
    def optimize(self, n_iter, trajectories):
        self.build_full(trajectories)
        for i in range(n_iter):
            self.propose(trajectories)
            if not self.compare():
                self.revert()


def is_better(lam, lam0):
    """Compares lists of ordered eigenvalues for slowness."""
    try:
        first_gain = np.where(lam > lam0)[0][0]
        first_loss = np.where(lam < lam0)[0][0]
        return first_gain < first_loss
    except:
        return False

def set_feature(X_list, Y_list, i):
    for k in range(len(X_list)):
        X_list[k][:, i] = Y_list[k][:, 0]
