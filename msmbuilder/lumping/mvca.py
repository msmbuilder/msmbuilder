# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: 
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
from msmbuilder.cluster import LandmarkAgglomerative
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster.agglomerative import pdist
from msmbuilder.utils.divergence import js_metric_array
import scipy.cluster.hierarchy


class MVCA(MarkovStateModel):
    """Minimum Variance Cluster Analysis (MVCA) for coarse-graining (lumping)
    microstates into macrostates.

    Parameters
    ----------
    n_macrostates : int or None
        The desired number of macrostates in the lumped model. If None,
        only the linkages are calcluated (see ``use_scipy``)
    metric : string or callable, default=js_metric_array
        Function to determine pairwise distances. Can be custom.
    n_landmarks : int, optional
        Memory-saving approximation. Instead of actually clustering every
        point, we instead select n_landmark points either randomly or by
        striding the data matrix (see ``landmark_strategy``). Then we cluster
        the only the landmarks, and then assign the remaining dataset based
        on distances to the landmarks. Note that n_landmarks=None is equivalent
        to using every point in the dataset as a landmark.
    landmark_strategy : {'stride', 'random'}, default='stride'
        Method for determining landmark points. Only matters when n_landmarks
        is not None. "stride" takes landmarks every n-th data point in X, and
        random selects them  uniformly at random.
    random_state : integer or numpy.RandomState, optional
        The generator used to select random landmarks. Only used if
        landmark_strategy=='random'. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Attributes
    ----------
    microstate_mapping_ : np.array, [number of microstates]

    Notes
    -----
    MVCA is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on MVCA refer to the MICROSTATE properties--e.g.
    mvca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of MVCA().
    MVCA will scale poorly with the number of microstates. Consider
    use_scipy=False and n_landmarks < number of microstates.
    """

    def __init__(self, n_macrostates, metric=js_metric_array, 
                 n_landmarks=None, landmark_strategy='stride',
                 random_state=None, **kwargs):
        self.n_macrostates = n_macrostates
        self.metric = metric
        self.n_landmarks = n_landmarks
        self.landmark_strategy = landmark_strategy
        self.random_state = random_state
        super(MVCA, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        """Fit a MVCA lumping model using a sequence of cluster assignments.

        Parameters
        ----------
        sequences : list(np.ndarray(dtype='int'))
            List of arrays of cluster assignments
        y : None
            Unused, present for sklearn compatibility only.

        Returns
        -------
        self
        """
        super(MVCA, self).fit(sequences, y=y)
        if self.n_macrostates is not None:
            self._do_lumping()
        else:
            raise RuntimeError('n_macrostates must not be None to fit')

        return self

    def _do_lumping(self):
        """Do the MVCA lumping.
        """
        model = LandmarkAgglomerative(linkage='ward',
                                      n_clusters=self.n_macrostates,
                                      metric=self.metric,
                                      n_landmarks=self.n_landmarks,
                                      landmark_strategy=self.landmark_strategy,
                                      random_state=self.random_state)
        microstate_mapping_ = model.fit_transform([self.transmat_])[0]

        self.microstate_mapping_ = microstate_mapping_

    def partial_transform(self, sequence, mode='clip'):
        if self.n_macrostates is None:
            raise RuntimeError('n_macrostates must not be None to transform')
        trimmed_sequence = super(MVCA, self).partial_transform(sequence, mode)
        if mode == 'clip':
            return [self.microstate_mapping_[seq] for seq in trimmed_sequence]
        elif mode == 'fill':
            def nan_get(x):
                try:
                    x = int(x)
                    return self.microstate_mapping_[x]
                except ValueError:
                    return np.nan

            return np.asarray([nan_get(x) for x in trimmed_sequence])
        else:
            raise ValueError

    @classmethod
    def from_msm(cls, msm, n_macrostates, metric=js_metric_array, 
                 n_landmarks=None, landmark_strategy='stride',
                 random_state=None, get_linkage=False):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates
        get_linkage : boolean, default=False
            Whether to return linkage and elbow data objects. Warning:
            This will compute n choose 2 pairwise distances

        Returns
        -------
        lumper : cls
            The fit MVCA object.
        pairwise_dists : if get_linkage is True, np.array,
                         [number of microstates choose 2]
        linkage : if get_linkage is True, scipy linkage object
        elbow_data : if get_linkage is True, np.array,
                     [number of microstates - 1]. Change in updated Ward
                     objective function, indexed by n_macrostates - 1

        Example
        -------
        plt.figure()
        scipy.cluster.hierarchy.dendrogram(mvca.linkage)

        scatter(arange(1,n_microstates), mvca.elbow_data)
        """
        params = msm.get_params()
        lumper = cls(n_macrostates, metric=metric, 
                 n_landmarks=n_landmarks, landmark_strategy=landmark_strategy,
                 random_state=random_state, **params)

        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        lumper.n_states_ = msm.n_states_

        if n_macrostates is not None:
            lumper._do_lumping()

        if get_linkage:
            p = pdist(msm.transmat_, metric=metric)
            l = scipy.cluster.hierarchy.linkage(p, 'ward')

            lumper.pairwise_dists = p
            lumper.linkage = l
            lumper.elbow_data = l[:, 2][::-1]

        else:
            lumper.pairwise_dists = None
            lumper.linkage = None
            lumper.elbow_data = None

        return lumper
