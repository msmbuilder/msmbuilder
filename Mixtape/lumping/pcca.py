import msmbuilder.lumping
import msmbuilder as msmb
from sklearn.base import BaseEstimator, TransformerMixin, clone
import mixtape
import numpy as np

class PCCA(mixtape.markovstatemodel.MarkovStateModel):
    """Perron Cluster Cluster Analysis (PCCA) for coarse-graining (lumping)
        microstates into macrostates.  This reference implementation uses MSMBuilder
        for the PCCA code but uses the Mixtape MarkovStateModel class for
        estimating the microstate transition matrix.
    
    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        mixtape.markovstatemodel.MarkovStateModel for possibile options.

    """

    def __init__(self, n_macrostates, **kwargs):
        self.n_macrostates = n_macrostates
        super(PCCA, self).__init__(**kwargs)

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
        
        super(PCCA, self).fit(sequences, y=y)
        self._pcca =  msmb.lumping.PCCA(self.transmat_, self.n_macrostates)
        return self

    @property
    def trimmed_microstates_to_macrostates(self):
        return dict((key, self._pcca.microstate_mapping[val]) for (key, val) in self.mapping_.iteritems())

    def transform(self, sequences):
        """Map microstates onto macrostates, performing trimming if necessary.

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
        trimmed_sequences = super(PCCA, self).transform(sequences)
        return [np.array(map(lambda x: self.trimmed_microstates_to_macrostates[x], seq)) for seq in trimmed_sequences]


class PCCAPlus(PCCA):
    """Perron Cluster Cluster Analysis Plus (PCCA+) for coarse-graining (lumping)
        microstates into macrostates.  This reference implementation uses MSMBuilder
        for the PCCA+ code but uses the Mixtape MarkovStateModel class for
        estimating the microstate transition matrix.
    
    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        mixtape.markovstatemodel.MarkovStateModel for possibile options.

    """

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
        
        super(PCCA, self).fit(sequences, y=y)        
        self._pcca =  msmb.lumping.PCCAPlus(self.transmat_, self.n_macrostates)
        return self
