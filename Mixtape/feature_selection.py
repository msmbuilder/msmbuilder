import sklearn, sklearn.base
import numpy as np
import mixtape.tica
import mixtape.subset_featurizer


def clone_and_swap(featurizer):
    """Clone a featurizer and randomly swap one of its features.
    
    Parameters
    ----------
    featurizer : SubsetUnionFeaturizer
        The featurizer to clone and swap.
    
    Returns
    -------
    featurizer : SubsetUnionFeaturizer
        A new featurizer with one of its features swapped with a randomly selected feature.
        
    ToDo
    ----
    Make a separate MCMC mover class, or make this function a class method
    on SubsetUnionFeaturizer?
    """
    featurizer = sklearn.clone(featurizer)
    new_feature = np.random.choice(featurizer.n_featurizers)
    new_value = np.random.choice(featurizer.n_max_i[new_feature])
    
    current_feature_weights = featurizer.n_features_i / (1.0 * featurizer.n_features_i.sum())
    
    featurizer_to_replace = np.random.choice(featurizer.n_featurizers, p=current_feature_weights)
    
    featurizer_index_to_replace = np.random.choice(featurizer.n_features_i[featurizer_to_replace])
    
    print("%s %d %d %d %d" % (" " * 80, featurizer_to_replace, featurizer_index_to_replace, new_feature, new_value))
    
    if featurizer_to_replace == new_feature and new_value == featurizer_index_to_replace:
        print('no swap')
        return featurizer
    
    if new_value in featurizer.transformer_list[new_feature][1].subset:
        print('no swap (already have feature)')
        return featurizer        
    
    value_to_remove = featurizer.transformer_list[featurizer_to_replace][1].subset[featurizer_index_to_replace]
    set0 = set(featurizer.transformer_list[featurizer_to_replace][1].subset)
    set0.remove(value_to_remove)
    featurizer.transformer_list[featurizer_to_replace][1].subset = np.array(list(set0))
    
    set1 = set(featurizer.transformer_list[new_feature][1].subset)
    set1.add(new_value)
    featurizer.transformer_list[new_feature][1].subset = np.array(list(set1))

    return featurizer


class Optimizer(sklearn.base.BaseEstimator):
    """Optimize objective function by swapping active features one-by-one.
    
    Parameters
    ----------
    featurizer : SubsetUnionFeaturizer
        The featurizer to clone and swap during optimization.
    model : BaseEstimator
        Some model object that implements score() that can be used to
        build and evaluate models.
    n_iter : int
        Number of iterations to attempt when optimize() is called.
    """
    def __init__(self, featurizer, model, n_iter):
        self.featurizer = featurizer
        self.model = model
        self.n_iter = n_iter
    
    def optimize(self, trajectories):
        """Optimize objective function by random swapping.
        
        Parameters
        ----------
        trajectories : list of md.Trajectory
            Trajectories to use.
        """
        
        features = self.featurizer.transform(trajectories)
        self.model = sklearn.clone(self.model)
        self.model.fit(features)
        self.current_score = self.model.score(features)
        
        for i in range(self.n_iter):
            new_featurizer = clone_and_swap(self.featurizer)
            
            features = new_featurizer.transform(trajectories)
            new_model = sklearn.clone(self.model)
            new_model.fit(features)
            new_score = new_model.score(features)

            if new_score > self.current_score:
                accept = True
            else:
                accept = False

            print("%d * %.4f %.4f" % (accept, self.current_score, new_score))
            
            if accept:
                self.model = new_model
                self.featurizer = new_featurizer
                self.current_score = new_score                
        
        return self.featurizer
