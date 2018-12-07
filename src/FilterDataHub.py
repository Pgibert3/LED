from OnsetStrengthFilter import OnsetStrengthFilter


class FilterDataHub:

    def __init__(self):
        # Default call OnsetStrengthFilter(44100, 512, 2500, 2048, 128, 4)
        self._osfilter = OnsetStrengthFilter(44100, 512, 2500, 2048, 128, 4)

    def get_next_os_all_bands(self):
        """
        Returns the most recent onset strength of the audio
        @return: onset strength subbands
        @rtype: np.array((?, 1), dtype=np.float32)  # ? is number of subbands, but this class doesn't know that number
        """
        return self._osfilter.next_os_all_bands()

    def get_next_onset(self):
        """
        Returns True if the most recent onset strength is a peak
        @return: an array of Booleans, each element true if os of subband is an onset
        @rtype: np.array(self._num_subbands, dtype=Boolean)
        """
        os_bands = self.get_next_os_all_bands()
        return self._osfilter.is_os_peak(os_bands)



