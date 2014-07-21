import msmbuilder.lumping
import msmbuilder as msmb
from sklearn.base import BaseEstimator, TransformerMixin, clone
import mixtape
import numpy as np

class PCCA(BaseEstimator, TransformerMixin):
    """Perron Cluster Cluster Analysis (PCCA) for coarse-graining (lumping)
        microstates into macrostates.  This reference implementation uses MSMBuilder
        for the PCCA code but uses the Mixtape MarkovStateModel class for
        estimating the microstate transition matrix.
    
    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    msm_model : mixtape.MarkovStateModel, optional
        An instantiated (but not fit) MSM (or compatible) object that
        will be used to build and estimate the Microstate transition matrix.
        If None, mixtape.MarkovStateModel() will be used with default parameters.
        The input MSM object will be cloned via `sklearn.clone()` to remove
        any parameters that have previously been learned via data.
    
    Attributes
    ----------
    cached_msm : mixtape.MarkovStateModel
        PCCA builds and caches a microstate MSM for estimating the transition
        matrix.

    """

    def __init__(self, n_macrostates, msm_model=None):
        self.n_macrostates = n_macrostates
        self.msm_model = msm_model

    def fit(self, sequences, y=None):
        """Fit a PCCA lumping model using a sequence of cluster assignments.

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
        
        self._build_msm(sequences)
        
        self._pcca =  msmb.lumping.PCCA(self._cached_msm.transmat_, self.n_macrostates)
        return self

    def _build_msm(self, sequences):
        """Build and cache a microstate MSM for estimating the transition matrix."""
        if self.msm_model is None:
            self._cached_msm = mixtape.markovstatemodel.MarkovStateModel()
        else:
            self._cached_msm = clone(self.msm_model)
        
        self._cached_msm.fit(sequences)

    @property
    def mapping_(self):
        return dict((key, self._pcca.microstate_mapping[val]) for (key, val) in self._cached_msm.mapping_.iteritems())

    def transform(self, sequences):
        """Map microstates onto macrostates

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
        micro_sequences = self._cached_msm.transform(sequences)  # Ergodic Trim
        return [np.array(map(lambda x: self.mapping_[x], seq)) for seq in micro_sequences]


class PCCAPlus(PCCA):
    """Perron Cluster Cluster Analysis Plus (PCCA+) for coarse-graining (lumping)
        microstates into macrostates.  This reference implementation uses MSMBuilder
        for the PCCA+ code but uses the Mixtape MarkovStateModel class for
        estimating the microstate transition matrix.
    
    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    
    Attributes
    ----------
    cached_msm : mixtape.MarkovStateModel
        PCCA+ builds and caches a microstate MSM for estimating the transition
        matrix.

    """

    def fit(self, sequences):
        """Fit a PCCA lumping model using a sequence of cluster assignments.

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
        
        self._build_msm(sequences)
        
        self._pcca =  msmb.lumping.PCCAPlus(self._cached_msm.transmat_, self.n_macrostates)
        return self
