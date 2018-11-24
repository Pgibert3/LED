import numpy as np
import pyaudio


class AUXBackend:

    def __init__(self, sr):
        """
        Uses the PyAudio library to open an audio stream t the local device/'s microphone or alternate auxiliary input
        @param sr: sampling rate
        @type sr: int
        """
        # TODO: Need to store hop_length and frame_size init values elsewhere
        HOPLENGTH = 512
        FRAMESIZE = 2048 # the size of the audio chunk being stored

        self._sr = sr
        self._hop_length = HOPLENGTH  # the number of new samples to read in per audio frame
        self._frame_size = FRAMESIZE  # the size of the audio chunk being stored
        self._frame = np.zeros(self._frame_size) # a chunk of audio that is appended to per audio read request

        # Open a PyAudio stream
        self._p = pyaudio.PyAudio()  # a PyAudio object. Stored to allow for proper termination of the audio stream
        # PyAudio stream of audio from the local device's microphone or alternate auxiliary input
        self._stream = self._p.open(
            input=True,  # sets the mode of the PyAudio stream to input/output
            format=pyaudio.paFloat32,  # read in data as a certain format
            channels=1,  # number of channels to read
            rate=self._sr,  # set the sampling rate of the audio stream
            frames_per_buffer=self._hop_length  # how many data points to read per read request
        )

    def next_frame(self):
        """
        Reads from the PyAudio stream and returns the resulting frame
        @return: a frame of audio
        @rtype: np.ndarray(self._frame_size, dtype=np.float32)
        """
        data = self._read_samples()
        frame = self._store_samples(data)
        return frame

    def _read_samples(self):
        """
        Reads samples from the PyAudio stream and returns the data
        @return: newly read audio ramples
        @rtype: np.ndarray(self._hop_length, dtype=np.float32)
        """
        samples = np.frombuffer(self._stream.read(
            self._hop_length,  # how many samples to read in
            exception_on_overflow=False  # Prevents weird errors with the PyAudio stream
        ), dtype=np.float32)
        return samples

    def _store_samples(self, samples):
        """
        Appends input to the frame, popping off len(input) data points from the beginning of the frame,
        and then returns the new frame
        @param samples: samples to append
        @type samples: np.ndarray(self._hop_length, dtype=np.float32)
        @return: the full audio frame currently stored in self._frame
        @rtype: np.ndarray(self._frame_size, dtype=np.float32)
        """
        self._frame = np.delete(self._frame, np.s_[:self._hop_length])
        self._frame = np.append(self._frame, samples)
        return self._frame

    def _stop_pyaudio_stream(self):
        """
        Terminates the stream by terminating port audio as recommended by PyAudio docs
        """
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()
