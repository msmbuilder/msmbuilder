class BaseSelector(object):
    def iterate(self):
        pass

    def propose(self):
        pass
        
    def build(self):
        pass

class SingleSwapSelector(BaseSelector):
    """Select features by random swaps of single features."""
    pass


class ScoreMixin(object):
    pass


class TICASelector(SingleSwapSelector, ScoreMixin):
    pass


class GeneralTICAOptimizer(object):
    def __init__(self, subset_featurizer, trj0):
        self.trj0 = trj0        
        self.subset_featurizer = subset_featurizer
        self.lam0 = None
    
    def build(self, trajectories):
        tica = mixtape.tica.tICA()
        unused_output = map(lambda trj: tica.partial_fit(self.subset_featurizer.featurize(trj)), trajectories)  # Weird linalg issues can occur
        return tica
    
    def revert(self):
        pass
        
    def optimize(self, n_iter, trajectories):
        
            
        for i in range(n_iter):
            self.propose()
            if self.compare(new_model, old_model):
                self.revert()
    
    def propose(self, trajectories):
        self.subset_featurizer.propose_swap()
        self.build(trajectories)
    
    def try_swap(self, trial_i, new_value, old_value, trajectories):
        
        if self.lam0 is None:  # Only on first round
            self.tica = self.build_tica(trajectories)
            self.lam0 = self.tica.eigenvalues_.copy()    
            
        self.subset_featurizer.swap(trial_i, new_value)

        self.tica = self.build_tica(trajectories)
        self.lam = self.tica.eigenvalues_.copy()
            
        if not is_better(self.lam, self.lam0):
            accept = False
            self.subset_featurizer.swap(trial_i, old_value)
        else:
            self.lam0 = self.lam
            accept = True
        print("%d swap %d (%.6d -> %.6d) %.5f %.4f %.4f %.5f %.4f %.4f" % (accept, trial_i, old_value, new_value, self.lam0[0], self.lam0[1], self.lam0[2], self.lam[0], self.lam[1], self.lam[2]))
        #print("%.5f %.4f" % (self.lam0[0], self.lam0[1]))
        #print("%.5f %.4f" % (self.lam[0], self.lam[1]))
