import numpy as np
from librosa import feature, onset, core
from InputInterface import InputInterface
from Exceptions import AudioStreamException
import ProcessingConfig as config


class FeatureExtractor:
    """Extracts onsets, ..., from audio input

	Attributes:
		input_settings -- the kwargs passed to InputInterface when preparing to stream data
		channels -- boundries of subbands to calculate oss of
		sr -- sampling rate
		hop_length -- new frames to read in to buffer for analysis using feature of onset_strength_multi
		n_fft -- total frames of buffer for analysis using feature of onset_strength_multi
		oss_buff_length -- length of oss_envelope stored for use in oss normalisation (frames)
	"""

	def __init__(self):
        self.channels = config.extractor_settings['channels']
        self.sr = config.extractor_settings['sr']
        self.hop_length = config.extractor_settings['hop_length']
        self.n_fft = config.extractor_settings['n_fft']
        self.oss_buff_length = config.extractor_settings['oss_buff_length']
        self.oss_env = np.zeros((self.oss_buff_length, 1 if self.channels is None else len(
            self.channels) - 1))  # Stored as a multi-demensional point for graphing

        self.input = None;

        # Tracked values for onset detection
        self.oss_min = [];  # Local min of oss_envelope per channel
        self.oss_max = [];  # Local max of oss_envelope per channel
        self.fslp = np.zeros(len(self.channels) - 1);  # Frames since last peak

    # Stores the provided array of onset strength signals in the oss_env
    def _store_oss(self, oss):
        self.oss_env = np.delete(self.oss_env, 0, axis=0)
        self.oss_env = np.append(self.oss_env, oss, axis=0)

    # Computes the oss of a frame
    def get_oss(self, frame, n_mels=128):
        oss = onset.onset_strength_multi(
            y=frame,
            channels=self.channels,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=n_mels  # specefic to melspectrogram feature
        )
        oss = (oss[:, 4])  # select correct data
        oss = np.reshape(oss, (1, oss.shape[0]))  # np.ndarray([oss ch1, oss ch2, ...])
        return oss

    # from oss, subtract min value of oss envelope, divide by envelope max and then divide by envelope std
    def _normalize_oss(self, oss):
        oss_min = np.min(self.oss_env, axis=0)
        oss_max = np.min(self.oss_env, axis=0)
        oss_std = np.std(self.oss_env, axis=0)
        # Filter for erroneous data
        oss_min = np.max([np.zeros(len(oss_min)), oss_min], axis=0)
        oss_max[oss_max <= 0] = 1
        oss_std[oss_std <= 0] = 1
        return ((oss - oss_min) / oss_max) / oss_std

    # Returns true if oss value represents a peak in the oss envelope. Assumes the oss being evaluated has been stored in the envelope. Implentation based on Librosa.util.peak_pic()
    def _peak_pick(
            self,
            oss,
            pre_max_c=0.03,
            pre_avg_c=0.1,
            wait_c=0.03,
            delta=.07):
        scale = self.sr // self.hop_length  # scales the kwargs values to frame lengths
        pre_max = self.oss_env[-int(pre_max_c * scale):][0]  # use index 0 to break out a bracket level
        pre_avg = self.oss_env[-int(pre_avg_c * scale):][0]  # use index 0 to break out a bracket level
        wait = int(wait_c * scale)
        is_peak = np.all([
            oss[0] == np.max(pre_max, axis=0),  # use index 0 to break out a bracket level
            oss[0] >= np.mean(pre_avg, axis=0) + delta,  # use index 0 to break out a bracket level
            self.fslp > wait
        ],
            axis=0)
        # Set the value of fslp based on whether a new peak has been detected
        self.fslp[is_peak] = 0
        self.fslp[is_peak == False] += 1
        return is_peak

    # Returns true if the given oss is an onset and stores the oss in the oss envelope. Implementation based on librosa.onset.onset_detect()
    def is_onset(self, oss):
        self._store_oss(oss)
        oss = self._normalize_oss(oss)
        return self._peak_pick(oss)  # Use default kwargs

    # Uses recent data to update the tempo
    def _update_tempo(self, frame):
        pass

    # Returns the current tempo
    def get_tempo(self):
        pass

    def open_audio_stream(self):
        if (self.input is None):
            self.input = InputInterface()
            try:
                self.input.open_stream()
            except AudioStreamException as e:
                print(e)
        else:
            raise AudioStreamException(
                "Cannot reopen a running audio stream",
                "FeatureExtractor.open_audio_stream()")

    def close_audio_stream(self):
        if (self.input is None):
            raise AudioStreamException(
                "There is no audio stream to close",
                "FeatureExtractor.close_audio_stream()")
        else:
            try:
                self.input.close_stream()
                self.input = None
            except AudioStreamException as e:
                print(e)
                sys.exit()

    def fetch_data(self):
        if (self.input is None):
            raise AudioStreamException("There is no running audio stream", "FeatureExtractor.fetch_data()")
        else:
            try:
                frame = self.input.get_next_frame()
                oss = self.get_oss(frame)
                is_onset = self.is_onset(oss)
                data = {
                    "oss": oss,
                    "is_onset": is_onset
                }
                return data
            except AudioStreamException as e:
                print(e)
                sys.exit()
