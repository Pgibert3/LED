import numpy as np
import warnings
import pyaudio
from librosa_z import util
import pdb

class Spectrum:
    def __init__(self, sr=22050, frame_size=2048, hop=512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop = hop

        self.S_ref = None
        self.yb = np.zeros(self.frame_size)
        self.pk_delay = 0
        self.start_delay = self.frame_size // self.hop

        self.hist_y = np.array([])
        self.hist_onset = np.array([0])


    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                format = pyaudio.paFloat32,
                channels = 1,
                rate = self.sr,
                input = True,
                frames_per_buffer = self.hop)


    def stop(self):
        try:
            self.p
        except AttributeError:
            warnings.warn('No audio stream detected. Consider calling Spectrum.start()')
            return

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def update_yb(self, y):
        self.yb = np.delete(self.yb, np.s_[:self.hop])
        self.yb = np.append(self.yb, y)


    def append_hist_y(self, y):
        self.hist_y = np.append(self.hist_y, y)


    def append_hist_onset(self, onset):
        self.hist_onset = np.append(self.hist_onset, onset)


    def melspectrogram(self, y, n_mels=128):
        window = np.hanning(len(y))
        S = np.fft.fft(y * window)
        frst_half = int(len(S)//2 + 1)
        S = np.abs(S[:frst_half])**2
        mel_basis = util.mel(sr=self.sr, n_fft=self.frame_size, n_mels=n_mels)

        return np.dot(mel_basis, S)


    def aggregate(self, y, a, b):
        #TODO: Add error checking
        return a * np.mean(y) + b * np.median(y)


    def onset_strength(self, y):
        S = self.melspectrogram(y)
        S = util.power_to_db(S)

        if self.S_ref is None:
            self.S_ref = S.copy()
        onset = S - self.S_ref
        onset = np.maximum(0.0, onset)
        onset = self.aggregate(onset, 1, 0)
        return onset, S


    def is_peak(self, onset, hist=30, wait=5, c=2.0):
        if len(self.hist_onset) + 1 < hist:
            return False
        else:
            prev = self.hist_onset[-(len(self.hist_onset)//3 + 1):-1]
            prev_max = np.max(prev)
            prev_avg = np.mean(prev)

            if self.pk_delay > 0:
                self.pk_delay -=1
                return False
            elif onset > prev_max and onset > prev_avg * c:
                self.pk_delay = wait
                return True


    def get_onset(self):
        return self.hist_onset[-1]


    def read_onset(self):
        try:
            y = np.fromstring(self.stream.read(self.hop, exception_on_overflow=False), dtype=np.float32)
        except AttributeError:
            warnings.warn('No audio stream detected. Consider calling Spectrum.start()')
            return

        self.update_yb(y)
        onset, self.S_ref = self.onset_strength(self.yb)
        self.append_hist_y(y)

        if self.start_delay > 0:
            self.append_hist_onset(0)
            self.start_delay -=1
        else:
            self.append_hist_onset(onset)

        return self.is_peak(onset)
