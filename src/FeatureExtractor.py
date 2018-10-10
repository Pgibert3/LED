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
		oss_buff_length -- length of oss_envelope stored for use in oss normalisation (frames)
	"""
	def __init__(self, channels, sr=44100, hop_length=512, n_fft=2048, oss_buff_length=2500):
		self.channels = channels
		self.sr = sr
		self.hop_length = hop_length
		self.n_fft = n_fft
		self.oss_buff_length = oss_buff_length
		self.oss_env = np.zeros((oss_buf_length, len(self.channels)-1)) #Stored as a multi-demensional point for graphing

		#Tracked values for onset detection
		self.oss_min = []; #Local min of oss_envelope per channel
		self.oss_max = []; #Local max of oss_envelope per channel
		self.fslp = 0; #Frames since last peak

	#Stores the provided array of onset strength signals in the oss_env
	def _store_oss(self, oss):
		self.oss_env = np.delete(self.oss_env, 0, axis=0)
		self.oss_env = np.append(self.oss_env, oss, axis=0)

	#Computes the oss of a frame and stores it
	def _oss(self, frame, n_mels=128):
		oss_env = onset.onset_strength_multi(
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

	#Returns true if oss value represents a peak in the oss envelope. Assumes the oss being evaluated has been stored in the envelope. Implentation based on Librosa.util.peak_pic()
	def _peak_pick(
			self,
			oss,
			pre_max_c=0.03,
			pre_avg_c=0.1,
			wait_c=0.03,
			delta=.07):
		scale = self.sr // self.hop_length #scales the kwargs values to frame lengths
		pre_max = self.oss_env[-pre_max_c*scale:]
		pre_avg = self.oss_env[-pre_avg_c*scale:]
		wait = wait_c*scale
		is_peak = (
			oss = np.max(pre_max, axis=0)
			&& oss >= np.mean(pre_avg, axis=0) + delta
			&& self.fslp > wait)
		#Set the value of wait based on whether a new peak has been detected
		if (is_peak):
			self.fslp = wait
		else:
			self.fslp -= 1
		return is_peak

	#Returns true if current frame is an onset and stores the oss in the oss envelope. Implimentation based on Librosa.onset.onset_detect()
	def is_onset(self, frame):
		oss = self._oss(frame)
		oss = self._normalize_oss(oss)
		self._store_oss(oss)
		return _self._peak_pick(oss) #Use default kwargs

	#Uses recent data to update the tempo
	def _update_tempo(self, frame):
		pass

	#Returns the current tempo
	def get_tempo(self):
		pass
