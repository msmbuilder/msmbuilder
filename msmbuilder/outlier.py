import numpy as np
from .cluster import MultiSequenceClusterMixin, BaseEstimator
from sklearn.svm import OneClassSVM

class OneClassSVMTrimmer(MultiSequenceClusterMixin, OneClassSVM, BaseEstimator):
    def partial_transform(self, traj):
        """Transform a single sequence based on outlier detection.

        Parameters
        ----------
        traj : sequence np.ndarray
            A time series sequence

        Returns
        -------
        trajout : np.ndarray, dtype=float, shape=(n_new, n_features)
            The output trajectory consists of the first `n_new` frames
            of the input, where `n_new` is the location of the first
            outlier in the sequence.  
        """
        yi = self.predict([traj])[0]
        try:
            ind = np.where(yi == -1)[0][0]  # Find the first outlier in the trajectory
        except IndexError:
            ind = None
        return traj[0:ind]

    def transform(self, traj_list, y=None):
        """Transform several sequences based on outlier detection

        Parameters
        ----------
        traj_list : list(mdtraj.Trajectory)
            List of A time series sequences

        Returns
        -------
        features : list(np.ndarray), length = len(traj_list)
            The output features will be truncated based on the first
            outlier detected in each timeseries.
        """
        return [self.partial_transform(traj) for traj in traj_list]
