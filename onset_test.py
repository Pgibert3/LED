import numpy as np
import matplotlib.pyplot as plt
import librosa
import Spectrum
import warnings
import pdb

spec = Spectrum.Spectrum()
peaks = []

def get_rec_time(y, sr, hop):
    return len(y) * hop / sr


def times(y, sr, hop):
    rec_time = get_rec_time(y, sr, hop)
    return np.linspace(0, rec_time, len(y))


def get_error(y, y_ref):
    if len(y) != len(y_ref):
        warnings.warn('y ({}) and y_ref ({}) must be equal in length'.format(\
                len(y), len(y_ref)))
        return '??'

    else:
        return '~' + str(round(np.mean(np.abs(y - y_ref)), 4)) + ' dB'

def plot_peaks(y, peaks, times, sub):
    plt.subplot(sub)
    plt.vlines(times[peaks], ymin=0, ymax=np.max(y) + 1, colors='r')


def plot_onset_env():
    rt_env = spec.hist_onset
    rosa_env = librosa.onset.onset_strength(y=spec.hist_y, sr=spec.sr)
    t_rt = times(spec.hist_onset, spec.sr, spec.hop)
    t_rosa = times(rosa_env, spec.sr, spec.hop)

    error = get_error(rt_env, rosa_env)

    plt.suptitle('Librosa vs Custom Real Time Onset Detection')
    plt.subplots_adjust(hspace=.5)

    plt.subplot(211)
    plt.plot(t_rosa, rosa_env)

    plt.title('Librosa Onset Envelope (REF)')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB)')
    plt.plot(t_rosa, rosa_env)

    plt.subplot(212)
    plt.title('Real Time Onset Envelope (Error: {})'.format(error))
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB)')
    plt.plot(t_rt, rt_env)

    plot_peaks(rt_env, peaks, t_rt, 212)

    plt.show()


def listen(time=None):
    if time is None:
        try:
            print('Started recording...')
            count = 0
            spec.start()
            while True:
                if spec.read_onset():
                    peaks.append(count)
                count += 1
        except KeyboardInterrupt:
            spec.stop()

    print('Recording haulted. Total recording time: {}s'\
            .format(round(get_rec_time(
                                    spec.hist_onset,
                                    spec.sr,
                                    spec.hop), 2)))
    plot_onset_env()
