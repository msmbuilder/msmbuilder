# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.
import numpy as np

from .base import MultiSequenceFeatureSelectionMixin


class FeatureSelector(MultiSequenceFeatureSelectionMixin):
    def __init__(self, features, which_feat=None):
        self.feats = dict(features)
        self.feat_list = list(self.feats)

        which_feat = which_feat if which_feat else self.feat_list[:]

        if not isinstance(which_feat, list):
            which_feat = [which_feat]
        elif not all([feat in self.feat_list for feat in which_feat]):
            raise ValueError('Not a valid feature')

        self.which_feat = which_feat

    def partial_transform(self, traj):
        return np.concatenate([self.feats[feat].partial_transform(traj)
                               for feat in self.which_feat], axis=1)
