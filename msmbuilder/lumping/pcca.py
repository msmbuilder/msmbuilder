from __future__ import print_function, division, absolute_import

import numpy as np
from ..msm import MarkovStateModel


class PCCA(MarkovStateModel):
    """Perron Cluster Cluster Analysis (PCCA) for coarse-graining (lumping)
    microstates into macrostates.

    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Notes
    -----
    PCCA is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on PCCA refer to the MICROSTATE properties--e.g.
    pcca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of PCCA().

    """

    def __init__(self, n_macrostates, pcca_tolerance=1e-5, **kwargs):
        self.n_macrostates = n_macrostates
        self.pcca_tolerance = pcca_tolerance
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
        """Do the PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing
            macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """

        # Extract non-perron eigenvectors
        right_eigenvectors = self.right_eigenvectors_[:, 1:]

        assert self.n_states_ > 0
        microstate_mapping = np.zeros(self.n_states_, dtype=int)

        def spread(x):
            return x.max() - x.min()

        for i in range(self.n_macrostates - 1):
            v = right_eigenvectors[:, i]
            all_spreads = np.array([spread(v[microstate_mapping == k])
                                    for k in range(i + 1)])
            state_to_split = np.argmax(all_spreads)
            inds = ((microstate_mapping == state_to_split) &
                    (v >= self.pcca_tolerance))
            microstate_mapping[inds] = i + 1

        self.microstate_mapping_ = microstate_mapping

    def partial_transform(self, sequence, mode='clip'):
        trimmed_sequence = super(PCCA, self).partial_transform(sequence, mode)
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
    def from_msm(cls, msm, n_macrostates):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
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
        lumper.n_states_ = msm.n_states_

        lumper._do_lumping()

        return lumper
