import numpy as np
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from pandas import DataFrame

from .base import MultiSequencePreprocessingMixin


class Butterworth(MultiSequencePreprocessingMixin):
    """Smooth time-series data using a low-pass, zero-delay Butterworth filter.

    Parameters
    ----------
    width : int, optional, default=5
        This acts very similar to the window size in a moving average
        smoother. In this implementation, the frequency of the low-pass
        filter is taken to be two over this width, so it's like
        "half the period" of the sinusiod where the filter starts
        to kick in. Must be an integer greater than one.
    order : int, optional, default=3
        The order of the filter. A small odd number is recommended. Higher
        order filters cutoff more quickly, but have worse numerical
        properties.

    References
    ----------
    .. [1] "FiltFilt". Scipy Cookbook. SciPy. <http://www.scipy.org/Cookbook/FiltFilt>.
    """

    def __init__(self, width=5, order=3, analog=True):

        if width < 2.0 or not isinstance(width, int):
            raise ValueError('width must be an integer greater than 1.')
        if not isinstance(order, int):
            raise ValueError('order must be an integer')

        # find nearest odd integer
        self.width = int(np.ceil((width + 1) / 2) * 2 - 1)

        self.order = order

        # Use lfilter_zi to choose the initial condition of the filter.
        self._num, self._denom = butter(self.order, 2.0 / self.width)
        self._zi = lfilter_zi(self._num, self._denom)

    def partial_transform(self, sequence):

        output = np.zeros_like(sequence)

        for i, x in enumerate(sequence.T):
            padded = np.r_[x[self.width - 1: 0: -1], x, x[-1: -self.width: -1]]

            # Apply the filter to the width.
            z, _ = lfilter(self._num, self._denom, padded,
                           zi=self._zi * padded[0])

            # Apply the filter again, to have a result filtered at an order
            # the same as filtfilt.
            z2, _ = lfilter(self._num, self._denom, z, zi=self._zi * z[0])

            # Use filtfilt to apply the filter.
            filtered = filtfilt(self._num, self._denom, padded)

            output[:, i] = filtered[(self.width - 1):-(self.width - 1)]

        return output


class EWMA(MultiSequencePreprocessingMixin):
    """Smooth time-series data using an exponentially-weighted moving average filter

    Parameters
    ----------
    com : float, optional
        Center of mass
    span : float, optional
        Specify decay in terms of span
    halflife : float, optional
        Specify decay in terms of halflife
    min_periods : int, default 0
        Number of observations in sample to require (only affects beginning)
    freq : None or string alias / date offset object, default=None
        Frequency to conform to before computing statistic time_rule is a
        legacy alias for freq
    adjust : boolean, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average)

    References
    ----------
    .. [1] "pandas.stats.moments.ewma". Pandas Documentation. Pandas. <http://pandas.pydata.org/pandas-docs/version/0.13.1/generated/pandas.stats.moments.ewma.html>.
    """
    def __init__(self, com=None, span=None, halflife=None, min_periods=0,
                 freq=None, adjust=True):
        self.com = com
        self.span = span
        self.halflife = halflife
        self.min_periods = min_periods
        self.freq = freq
        self.adjust = adjust

    def _ewma(self, sequence):
        ewma = DataFrame(sequence).ewm(com=self.com,
                                       span=self.span,
                                       halflife=self.halflife,
                                       min_periods=self.min_periods,
                                       freq=self.freq,
                                       adjust=self.adjust).mean()

        return ewma.values

    def partial_transform(self, sequence):
        return self._ewma(sequence)


class DoubleEWMA(EWMA):
    """Smooth time-series data using forward and backward exponentially-weighted moving average filters

    Parameters
    ----------
    com : float, optional
        Center of mass
    span : float, optional
        Specify decay in terms of span
    halflife : float, optional
        Specify decay in terms of halflife
    min_periods : int, default 0
        Number of observations in sample to require (only affects beginning)
    freq : None or string alias / date offset object, default=None
        Frequency to conform to before computing statistic time_rule is a
        legacy alias for freq
    adjust : boolean, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average)

    References
    ----------
    .. [1] "Smoothing with Exponentionally Weighted Moving Averages". Connor Johnson. <http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/>.
    """
    def partial_transform(self, sequence):
        fwd = self._ewma(sequence)
        bwd = self._ewma(sequence[::-1, :])

        return (fwd + bwd) / 2.
