from .base import MultiSequenceFeatureSelectionMixin


class FeatureSelector(MultiSequenceFeatureSelectionMixin):
    def __init__(self, features, which_feat=None):
        self.feats = dict(features)
        self.which_feat = (which_feat if which_feat is not None
                           else list(self.feats)[0])

    def partial_transform(self, traj):
        return self.feats[self.which_feat].partial_transform(traj)
