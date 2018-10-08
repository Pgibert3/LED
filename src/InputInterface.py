import numpy as np
import pyaudio
from librosa import amplitude_to_db

from Exceptions import AudioStreamException

class InputInterface:
	"""Interfaces the pyaudio object and stream to read in raw audio data

	Attributes:
		sr -- sample rate of input
		hop_length -- number of new samples to read in
		frame -- buffer for frame
		p -- pyaudio object for audio input
		stream -- auydio stream
	"""
	def __init__(self, sr=44100, frame_size=2048, hop_length=512):
		self.sr = sr
		self.hop_length = hop_length
		self.frame = np.zeros(frame_size)
		self.p = pyaudio.PyAudio()
		self.stream = None

	#Starts the audio stream
	def start_stream(self):
		if (self.stream is None):
			self.stream = self.p.open(
					input = True, #set mode of stream to input
					format = pyaudio.paFloat32, #read in data as a float32
					channels = 1, #read all data into a single channel
					rate = self.sr,
					frames_per_buffer = self.hop_length
			)
		else:
			raise AudioStreamException(
					"Cannot run two audio streams",
					"start_stream()"
			)

	#Terminates the stream by terminating port audio and setting stream and p to None
	def stop_stream(self):
		if (self.stream is not None):
			self.stream.stop_stream()
			self.stream.close()
			self.p.terminate() #terminates port audio
		self.p = None
		self.stream = None

	#Stores new audio data into the frame buffer
	def _store_y(self, y):
		self.frame = np.delete(self.frame, np.s_[:self.hop_length]) #delte [self.hop_length] samples from the begining of frame
		self.frame = np.append(self.frame, y) # append [self.hop_length] samples to frame

	#reads in audio and returns the next frame
	def get_next_frame(self):
		if (self.stream is not None):
			#read in raw audio and format to np.float32
			y = np.frombuffer(self.stream.read(
					self.hop_length, #how many samples to read in
					exception_on_overflow=False #Prevents wierd errors with the audio stream
				), dtype=np.float32)
			self._store_y(y)
			return self.frame
		else:
			raise AudioStreamException(
					"There is not a running audio stream",
					"get_next_frame()"
			)
