import numpy as np
from librosa import feature, onset, core
from InputInterface import InputInterface

class FeatureExtractor:
	"""Extracts onsets, ..., from audio input

	Attributes:
		channels -- boundries of subbands to calculate oss of
		sr -- sampling rate
		hop_length -- new frames to read in to buffer for analysis using feature of onset_strength_multi
		n_fft -- total frames of buffer for analysis using feature of onset_strength_multi
		oss_buff_length -- length of oss_buffer stored for use in oss normalisation (frames)
	"""
	def __init__(self, channels, sr=44100, hop_length=512, n_fft=2048, oss_buff_length=2500):
		self.channels = channels
		self.sr = sr
		self.hop_length = hop_length
		self.n_fft = n_fft
		self.oss_buff_length = oss_buff_length
		self.oss_channels = np.zeros((oss_buf_length, len(self.channels)-1)) #Stored as a multi-demensional point for graphing

		#Tracked values for onset detection
		self.oss_min = []; #Local min of oss_buffer per channel
		self.oss_max = []; #Local max of oss_buffer per channel

	#Stores the provided array of onset strength signals in the oss_channels buffer
	def _store_oss(self, oss):
		self.oss_channels = np.delete(self.oss_channels, 0, axis=0)
		self.oss_channels = np.append(self.oss_channels, oss, axis=0)

	#Computes the oss of a frame and stores it
	def _oss(self, frame, n_mels=128):
		oss_channels = onset.onset_strength_multi(
				y=frame,
				channels=self.channels,
				sr=self.sr,
				hop_length=self.hop_length,
				n_fft=self.n_fft,
				n_mels=n_mels #specefic to melspectrogram feature
				)
		oss = (oss[:,4]) #select correct data
		oss = np.reshape(oss, (1, oss.shape[0])) #np.ndarray([oss ch1, oss ch2, ...])
		return oss

	#Sets the value of self.oss_min and self.oss_max based on new data and returns normalized oss
	def _normalize_oss(self, oss):
		self.oss_min = np.min((self.oss_min, oss), axis=0)
		self.oss_max = np.min((self.oss_min, oss), axis=0)
		return (oss - self.oss_min) / self.oss_max

	#Returns true if current frame is an onset. Implimentation based on Librosa.onset.onset_detect()
	def is_onset(self, frame):
		oss = self._oss(frame)
		oss = self._normalize_oss(oss)
		#TODO keep going

	#Uses recent data to update the tempo
	def _update_tempo(self, frame):
		pass

	#Returns the current tempo
	def get_tempo(self):
		pass
