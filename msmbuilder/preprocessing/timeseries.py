from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

from .base import MultiSequencePreprocessingMixin

class Butterworth(MultiSequencePreprocessingMixin):

    def __init__(self, width, order=3):
        """Smooth time-series data using a zero-delay Butterworth filter.

        Parameters
        ----------
        width : int
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


        if width < 2.0 or not isinstance(width, int):
            raise ValueError('width must be an integer greater than 1.')

        # find nearest odd integer
        self.width = int(np.ceil((width + 1) / 2) * 2 - 1)

        self.order = order

        # Use lfilter_zi to choose the initial condition of the filter.
        self._num, self._denom =  butter(self.order, 2.0 / self.pad)
        self._zi = lfilter_zi(b, a)

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

            output[:, i] = filtered[(self.width - 1) : -(self.width - 1)]

        return output
