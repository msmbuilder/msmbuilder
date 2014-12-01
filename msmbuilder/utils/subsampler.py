from __future__ import print_function, division, absolute_import
from sklearn.base import TransformerMixin
from ..base import BaseEstimator

__all__ = ['Subsampler']


class Subsampler(BaseEstimator, TransformerMixin):
    """Convert a list of feature time series (`X_all`) into a `lag_time`
    subsampled time series.

    Parameters
    ----------
    lag_time : int
        The lag time to subsample by
    sliding_window : bool, default=True
        If True, each time series is transformed into `lag_time` interlaced
        sliding-window (not statistically independent) sequences.  If
        False, each time series is transformed into a single subsampled
        time series.
    """
    def __init__(self, lag_time, sliding_window=True):
        self._lag_time = lag_time
        self._sliding_window = sliding_window

    def fit(self, X_all, y=None):
        return self

    def transform(self, X_all, y=None):
        """Subsample several time series.

        Parameters
        ----------
        X_all : list(np.ndarray)
            List of feature time series

        Returns
        -------
        features : list(np.ndarray), length = len(X_all)
            The subsampled trajectories.
        """
        if self._sliding_window:
            return [X[k::self._lag_time] for k in range(self._lag_time) for X in X_all]
        else:
            return [X[::self._lag_time] for X in X_all]
