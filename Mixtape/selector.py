import numpy as np
import mixtape.tica


class TICAScoreMixin(object):
    def compare(self):
        return is_better(self.model.eigenvalues_, self.old_model.eigenvalues_)

    def summarize(self):
        print("%d %.5f %.4f %.4f **** %.5f %.4f %.4f" % (
        self.accept, self.old_model.eigenvalues_[0], self.old_model.eigenvalues_[1], self.old_model.eigenvalues_[2], self.model.eigenvalues_[0], self.model.eigenvalues_[1], self.model.eigenvalues_[2]))


class TICAOptimizer(TICAScoreMixin):
    def __init__(self, subset_featurizer, trajectories):
        self.trj0 = trajectories[0][0]
        
        self.subset_featurizer = subset_featurizer
        
        self.old_subset = self.subset_featurizer.subset
        self.old_model, self.old_features = self.build(trajectories)
        
    
    def build(self, trajectories):
        tica = mixtape.tica.tICA()
        features = map(lambda trj: self.subset_featurizer.featurize(trj), trajectories)
        unused_output = map(lambda X: tica.partial_fit(X), features)
        return tica, features
    
    def revert(self):
        self.model, self.features = self.old_model, self.old_features
        self.subset_featurizer.set(self.old_subset)
        
    def propose(self, trajectories):
        self.to_replace, self.new_value, self.old_value, self.subset, self.old_subset = self.subset_featurizer.propose()
        self.subset_featurizer.set(self.subset)
        self.model, self.features = self.build(trajectories)
        
    def optimize(self, n_iter, trajectories):
        for i in range(n_iter):
            self.propose(trajectories)
            if not self.compare():
                self.accept =  False
                self.revert()
            else:
                self.accept =  True
            self.summarize()
            self.old_model, self.old_features, self.old_subset = self.model, self.features, self.subset



def is_better(lam, lam0):
    """Compares lists of ordered eigenvalues for slowness."""
    try:
        first_gain = np.where(lam > lam0)[0][0]
        first_loss = np.where(lam < lam0)[0][0]
        return first_gain < first_loss
    except:
        return False
