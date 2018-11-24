import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from InputInterface import InputInterface
from FeatureExtractor import FeatureExtractor
from FeatureDataBridge import FeatureDataBridge
import ProcessingConfig
from Exceptions import AudioStreamException

"""Script for viewing onset data

"""

#global project settings
sr = ProcessingConfig.project["SAMPLE_RATE"]

#input settings
input_hl = ProcessingConfig.input["HOP_LENGTH"]
frame_size = ProcessingConfig.input["FRAME_SIZE"]

#extractor settings
n_fft = ProcessingConfig.extractor["N_FFT"]
extractor_hl = ProcessingConfig.extractor["HOP_LENGTH"]
channels = ProcessingConfig.extractor["OSS_CHANNELS"]
oss_buff_length = ProcessingConfig.extractor["OSS_BUFF_LENGTH"]

#Intialize main objects
fdb = FeatureDataBridge()

N_OSS_CHANNELS = len(channels) - 1

#Start background processes
def start():
	input.start_stream()

#Stop background processes
def stop():
	input.stop_stream()

#Adds oss to oss buffer
def append_oss_buffer(oss, oss_buffer):
	print(oss)
	print(oss_buffer)
	return np.append(oss_buffer, oss, axis=0)

def get_time_axis(oss_buffer_length, hop_length, sr):
	tot_time = (oss_buffer_length * hop_length) / sr
	return np.linspace(0, tot_time, oss_buffer_length)

def plot_oss_channels(oss_buffer_length, hop_length, sr):
	times = get_time_axis(oss_buffer_length, hop_length, sr)
	plt.plot(times, oss_buffer, label='Inline label')
	plt.show()

if(__name__ == '__main__'):
	oss_buffer = np.zeros((1, N_OSS_CHANNELS))
	try:
		print("Recording audio...")
		while(True):
			data = fdb.get_feature_data()
			oss = data['oss']
			oss_buffer = append_oss_buffer(oss, oss_buffer)
	except KeyboardInterrupt:
		print("Done.")
		fdb.close()
		oss_buffer = np.delete(oss_buffer, 0, axis=0) #remove default value
		plot_oss_channels(len(oss_buffer), extractor_hl, 44100)
