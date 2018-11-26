import numpy as np
from librosa import onset
from AUXBackend import AUXBackend  # TODO: Setup project structure in PyCharm


class OnsetStrengthFilter:
    """     Subbands are stored as a multi-demensional point for graphing
            ---            ---
            | a0  a1  a2  .. | <- subband "a" time series is stored in row 0
            | b0  b1  b2  .. |
            | c0  c1  c2  .. |
            | d0  d1  d2  .. |
            | e0  e1  e2  .. |
            | :   :   :  .   |
            | :   :   :    . |
            ---            ---

            if channels is None, os_buffer defaults to a single channel
            TODO: num_subbands must be > 0. This condition is not checked for
    """

    def __init__(self, sr, hop_length, os_buffer_size, n_fft, n_mels, num_subbands):
        """
        Takes audio data from a backend source and converts it to onset strength data
        @param sr: sampling rate
        @type sr: int
        @param hop_length: number of new onset strength frames to read into the onset strength buffer
        @type hop_length: int
        @param os_buffer_size: size of the onset strength buffer
        @type os_buffer_size: int
        @param n_fft: number of bins to calculate when computing fft
        @type n_fft: int
        @param n_mels: number of bins to compute when calculating melspectrogram
        @type n_mels: int
        @param num_subbands: final subband count of onset strength data computed
        @type num_subbands: int
        """
        self._sr = sr
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._subbands = self._get_subband_boundaries(n_mels, num_subbands)

        self._backend = AUXBackend(self._sr)  # Object for getting raw audio input
        self._os_buffer = np.zeros((os_buffer_size, num_subbands))

        # Tracked values for onset detection
        self._os_min = []  # Local min of oss_envelope per channel
        self._os_max = []  # Local max of oss_envelope per channel

    def next_os_all_bands(self):
        """
        Gets an audio frame from the backend, calculates the onset strength subbands, then stores and returns the result
        @return: onset strength subbands
        @rtype: np.array((self._num_subbands, 1), dtype=np.float32)
        """
        frame = self._backend.next_frame()
        os = self._get_os(frame, self._n_mels)
        self._store_os(os)
        return os

    def _store_os(self, os):
        """
        Stores the provided array of onset strength signals in the oss_buffer
        @param os: a frame of onset strength subbands
        @type os: np.array((self._num_subbands, 1), dtype=np.float32)
        """
        self._os_buffer = np.delete(self._os_buffer, 0, axis=0)
        self._os_buffer = np.append(self._os_buffer, os, axis=0)

    def _get_os(self, frame, n_mels):
        """
        Computes the onset strength subbands of a frame
        @param frame: an audio frame
        @type frame: nd.array(?) TODO: What is this length?
        @param n_mels: number of subbands to compute
        @type n_mels: int
        @return: onset strength subbands
        @rtype: np.array((self._num_subbands, 1), dtype=np.float32)
        """
        os = onset.onset_strength_multi(
            y=frame,
            channels=self._subbands,
            sr=self._sr,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            n_mels=n_mels
        )
        os = (os[:, 4])  # TODO: Select the correct data here
        os = np.reshape(os, (1, os.shape[0]))
        return os

    def _normalize_os(self, os):
        """
        From os, subtract min value of oss envelope, divide by envelope max, and then divide by envelope std
        @param os: onset strength subbands
        @type os: np.array((self._num_subbands, 1), dtype=np.float32)
        @return: normalized onset strength subbands
        @rtype: nd.array((self._num_subbands, 1), dtype=np.float32)
        """
        os_min = np.min(self._os_buffer, axis=0)
        os_max = np.min(self._os_buffer, axis=0)
        os_std = np.std(self._os_buffer, axis=0)
        # Filter for erroneous data
        os_min = np.max([np.zeros(len(os_min)), os_min], axis=0)
        os_max[os_max <= 0] = 1
        os_std[os_std <= 0] = 1
        return ((os - os_min) / os_max) / os_std

    def _get_subband_boundaries(self, n_mels, num_subbands):
        """
        Returns the boundaries for the onset strength subbands
        @param n_mels: number of subbands to compute
        @type n_mels: int
        @param num_subbands: number of subbands per onset strength frame
        @type num_subbands: int
        @return: array conatining the boundary limits of each onset strength subband
        @rtype: nd.array(self._num_subbands + 1, dtype=int)
        """
        subbands = []
        step = n_mels / num_subbands
        for i in range(0, num_subbands + 1):
            if i == 0:
                subbands.append(0)
            else:
                sub = subbands[i - 1] + step
                subbands.append(int(sub))
        return subbands
