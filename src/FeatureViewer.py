import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from InputInterface import InputInterface
from FeatureExtractor import FeatureExtractor
import ProcessingConfig

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
input = InputInterface(sr=sr, frame_size=frame_size, hop_length=input_hl)
ext = FeatureExtractor(channels, sr=sr, hop_length=extractor_hl, n_fft=n_fft, oss_buff_length=oss_buff_length)

N_OSS_CHANNELS = len(channels) - 1

#Start background processes
def start():
    input.start_stream()

#Stop background processes
def stop():
    input.stop_stream()

#Adds oss to oss buffer
def append_oss_buffer(oss, oss_buffer):
    return np.append(oss_buffer, oss, axis=0)

#Gets the next channels by calling get_next_frame of InputInterface object
def next_oss_channels():
    try:
        frame = input.get_next_frame()
        oss = ext._oss(frame)
        return oss
    except Exception as e:
        print(e)
        sys.exit(0)

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
        print("Starting audio stream...")
        start()
        print("Done.")
    except Exception as e:
        print(e)
        stop()
        sys.exit(0)
    try:
        print("Recording audio...")
        while(True):
            oss = next_oss_channels()
            oss_buffer = append_oss_buffer(oss, oss_buffer)
    except KeyboardInterrupt:
        print("Done.")
        oss_buffer = np.delete(oss_buffer, 0, axis=0) #remove default value
        plot_oss_channels(len(oss_buffer), ext.hop_length, ext.sr)
        stop()
