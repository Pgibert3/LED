import librosa
from librosa import feature, display
import pyaudio
import numpy as np
import matplotlib.pyplot as plt


#Util functions

#Returns x axis in seconds of n audio samples
def calc_times(sr, samples):
    total_time = samples / sr
    return np.linspace(0,total_time, samples)

#Audio Varaibles
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
BUFFER = 2048
FEATURE = 'centroid'

#Setup audio stream
p = pyaudio.PyAudio()

stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=BUFFER
        )

#While loop to collect audio data
aud_data = np.array([])

print("Reading audio...")
while(True):
    try:
        #convert audio data into float
        y = np.fromstring(stream.read(BUFFER, exception_on_overflow=False))
        aud_data = np.append(aud_data, y)

    except KeyboardInterrupt:
        #Stop loop and stream
        print("Haulting audio stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Hault successful.")
        break


if FEATURE == 'chroma':
    chroma = feature.chroma_stft(y=aud_data, sr=RATE)
    #Plot data
    plt.figure(figsize=(10,4))
    display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()

elif FEATURE == 'centroid':
    cent = feature.spectral_centroid(y=aud_data, sr=RATE)
    D = np.abs(librosa.stft(aud_data))
    #Plot data
    plt.figure()
    plt.subplot(2,1,1)
    #plt.semilogy(cent.T, label='Spectral centroid')
    plt.plot(cent.T)
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, cent.shape[-1]])
    plt.legend()
    plt.subplot(2,1,2)
    display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('log Power spectrogram')
    plt.tight_layout()
    plt.show()
