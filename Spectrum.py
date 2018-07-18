import numpy as np
import warnings
import pyaudio
from librosa_z import util
import time
import color_switch
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, anim_conn, brain_conn sr=22050, frame_size=2048, hop=512):
        self.canim_conn = anim_conn
        self.brain_conn = brain_conn

        self.sr = sr
        self.frame_size = frame_size
        self.hop = hop

        self.S_ref = None
        self.yb = np.zeros(self.frame_size)
        self.pk_delay = 0
        self.start_delay = self.frame_size // self.hop
        self.norm = 1
        self.norm_count = self.sr * 3

        self.hist_y = np.array([])
        self.hist_onset = np.zeros(10)
        
        self.mel_basis = util.mel(sr=self.sr, n_fft=self.frame_size, n_mels=128)
        

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                format = pyaudio.paFloat32,
                channels = 1,
                rate = self.sr,
                input = True,
                frames_per_buffer = self.hop)
        self.run_onset_detection()


    def stop(self):
        try:
            self.p
        except AttributeError:
            warnings.warn('No audio stream detected.' +
                            ' Consider calling Spectrum.start()')
            return

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def update_yb(self, y):
        self.yb = np.delete(self.yb, np.s_[:self.hop])
        self.yb = np.append(self.yb, y)

    
    def append_hist_y(self, y):
        self.hist_y = np.append(self.hist_y, y)


    def update_hist_onset(self, onset):
        self.hist_onset = np.delete(self.hist_onset, 0)
        self.hist_onset = np.append(self.hist_onset, onset)
        

    def melspectrogram(self, y, n_mels=128):
        window = np.hanning(len(y))
        S = np.fft.fft(y * window)
        frst_half = int(len(S)//2 + 1)
        S = np.abs(S[:frst_half]/len(S))**2
        mspec = np.dot(self.mel_basis, S)

        return mspec


    def aggregate(self, y, a, b):
        #TODO: Add error checking
        return a * np.mean(y) + b * np.median(y)


    def onset_strength(self, y):
        y_h, y_p = util.hpss(y)
        S = self.melspectrogram(y)
        S = util.power_to_db(S)

        if self.S_ref is None:
            self.S_ref = S.copy()
        onset = S - self.S_ref
        onset = np.maximum(0.0, onset)
        onset = self.aggregate(onset, 1, 0)
        return onset, S

    #TODO: is_peak() needs further tuning
    def is_peak(self, onset, hist=6, wait=1, th=.7):
        if len(self.hist_onset) + 1 < hist:
            return False
        else:
            prev_max = np.max(self.hist_onset[-(hist//3):-1])
            prev_avg = np.mean(self.hist_onset[-(hist):-1])

            if self.pk_delay > 0:
                self.pk_delay -=1
                return False
            elif onset > prev_max and onset > prev_avg + th:
                self.pk_delay = wait
                return True


    def get_onset(self):
        return self.hist_onset[-1]


    def run_onset_detection(self):
        while True:
            try:
                y = np.fromstring(self.stream.read(self.hop, exception_on_overflow=False), dtype=np.float32)
            except AttributeError:
                warnings.warn('No audio stream detected. Consider calling Spectrum.start()')
                return

            self.update_yb(y)
            onset, self.S_ref = self.onset_strength(self.yb)
            #self.append_hist_y(y)

            if self.start_delay > 0:
                    self.update_hist_onset(0)
                    self.start_delay -= 1
            else:
                    self.update_hist_onset(onset)
            peak = self.is_peak(onset)
            self.send_data((onset, peak))
    
    
    def send_data(self, data):
        self.anim_conn.send(data)
        self.brain_conn.send(data)
             