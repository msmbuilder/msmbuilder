import msmbuilder.lumping
import msmbuilder as msmb
from sklearn.base import BaseEstimator, TransformerMixin, clone
from mixtape.markovstatemodel import MarkovStateModel
import numpy as np

class PCCA(MarkovStateModel):
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
    
    Notes
    -----
    PCCA is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on PCCA refer to the MICROSTATE properties--e.g. 
    pcca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of PCCA().  

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
        self._do_lumping()
        return self

    def _do_lumping(self):
        """Does the actual lumping part, change this function in subclasses."""
        self._pcca =  msmb.lumping.PCCA(self.transmat_, self.n_macrostates)

    @property
    def trimmed_microstates_to_macrostates(self):
        return dict((key, self._pcca.microstate_mapping[val]) for (key, val) in self.mapping_.items())

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
        f = np.vectorize(self.trimmed_microstates_to_macrostates.get)
        return [f(seq) for seq in trimmed_sequences]

    @classmethod
    def from_msm(cls, msm, n_macrostates):
        """Create and fit lumped model from pre-existing MSM.
        
        Parameters
        ----------
        msm : Mixtape.markovstatemodel.MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates
        
        Returns
        -------
        lumper : cls
            The fit PCCA(+) object.
        """
        
        params = msm.get_params()
        lumper = cls(n_macrostates, **params)
        
        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        
        lumper._do_lumping()
        
        return lumper
        

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
        mixtape.markovstatemodel.MarkovStateModel for possible options.

    Notes
    -----
    PCCAPlus is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on PCCAPlus refer to the MICROSTATE properties--e.g. 
    pcca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of PCCAPlus().  

    """
    def _do_lumping(self):
        self._pcca =  msmb.lumping.PCCAPlus(self.transmat_, self.n_macrostates)
