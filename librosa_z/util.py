import numpy as np
import warnings
from . import exceptions
import six
from scipy.ndimage import median_filter

'''included methods:
        power_to_db,
        db_to_power,
        amplitude_to_db,
        db_to_amplitude,
        hz_to_mel,
        mel_to_hz,
        mel_frequencies,
        fft_frequencies,
        mel,
        frames_to_samples,
        samples_to_frames,
        frames_to_time,
        time_to_frames,
        time_to_samples,
        samples_to_time,
        localmax,
        normalize,
        frame,
        autocorrelate,
        tempo_frequencies
        hpss
        softmask
        magphase
'''

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db   : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()

    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(magphase(D, power=2)[0]) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def db_to_power(S_db, ref=1.0):
    '''Convert a dB-scale spectrogram to a power spectrogram.

    This effectively inverts `power_to_db`:

        `db_to_power(S_db) ~= ref * 10.0**(S_db / 10)`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return ref * np.power(10.0, 0.1 * S_db)


def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    '''Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2)``, but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `20 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `S` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(20 * log10(S)) - top_db``


    Returns
    -------
    S_db : np.ndarray
        ``S`` measured in dB

    See Also
    --------
    power_to_db, db_to_amplitude

    Notes
    -----
    This function caches at level 30.
    '''

    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('amplitude_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call amplitude_to_db(magphase(D)[0]) instead.')

    magnitude = np.abs(S)

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value**2, amin=amin**2,
                       top_db=top_db)


def db_to_amplitude(S_db, ref=1.0):
    '''Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`:

        `db_to_amplitude(S_db) ~= 10.0**(0.5 * (S_db + log10(ref)/10))`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref: number > 0
        Optional reference power.

    Returns
    -------
    S : np.ndarray
        Linear magnitude spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return db_to_power(S_db, ref=ref**2)**0.5


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [1]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [2]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [3]_ (HTK) according to the following formula:

    `mel = 2595.0 * np.log10(1.0 + f / 700.0).`

    The choice of implementation is determined by the `htk` keyword argument: setting
    `htk=False` leads to the Auditory toolbox implementation, whereas setting it `htk=True`
    leads to the HTK implementation.

    .. [1] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [2] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [3] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.


    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc


    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.

    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).

    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).

    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices.

    Parameters
    ----------
    frames     : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number:
        `times[i] = frames[i] * hop_length`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y, sr=sr)
    >>> beat_samples = librosa.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def samples_to_frames(samples, hop_length=512, n_fft=None):
    """Converts sample indices into STFT frames.

    Examples
    --------
    >>> # Get the frame numbers for every 256 samples
    >>> librosa.samples_to_frames(np.arange(0, 22050, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : int or np.ndarray [shape=(n,)]
        sample index or vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : int or np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( samples[i] / hop_length )`

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)
    return np.floor((samples - offset) // hop_length).astype(int)


def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds).

    Parameters
    ----------
    frames     : np.ndarray [shape=(n,)]
        frame index or vector of frame indices

    sr         : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number:
        `times[i] = frames[i] * hop_length / sr`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y, sr=sr)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr)
    """

    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


def time_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    """Converts time stamps into STFT frames.

    Parameters
    ----------
    times : np.ndarray [shape=(n,)]
        time (in seconds) or vector of time values

    sr : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( times[i] * sr / hop_length )`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get the frame numbers for every 100ms

    >>> librosa.time_to_frames(np.arange(0, 1, 0.1),
    ...                         sr=22050, hop_length=512)
    array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    """

    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


def time_to_samples(times, sr=22050):
    '''Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : number or np.ndarray
        Time value or array of time values (in seconds)

    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : int or np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in `times`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    >>> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])

    '''

    return (np.asanyarray(times) * sr).astype(int)


def samples_to_time(samples, sr=22050):
    '''Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Sample index or array of sample indices

    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape, dtype=int]
        Time values corresponding to `samples` (in seconds)

    See Also
    --------
    samples_to_frames : convert sample indices to frame indices
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get timestamps corresponding to every 512 samples

    >>> librosa.samples_to_time(np.arange(0, 22050, 512))
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    '''

    return np.asanyarray(samples) / float(sr)


def localmax(x, axis=0):
    """Find local maxima in an array `x`.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmax(x, axis=0)
    array([[False, False, False],
           [ True, False, False],
           [False,  True,  True]], dtype=bool)
    >>> librosa.util.localmax(x, axis=1)
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True]], dtype=bool)

    Parameters
    ----------
    x     : np.ndarray [shape=(d1,d2,...)]
      input vector or array

    axis : int
      axis along which to compute local maximality

    Returns
    -------
    m     : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along `axis`

    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    '''Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that

        `norm(S, axis=axis) == 1`

    For example, `axis=0` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, `axis=1` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    `threshold` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.


    Parameters
    ----------
    S : np.ndarray
        The matrix to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : mininum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least `threshold` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of `S.dtype`.

    fill : None or bool
        If None, then columns (or rows) with norm below `threshold`
        are left as is.

        If False, then columns (rows) with norm below `threshold`
        are set to 0.

        If True, then columns (rows) with norm below `threshold`
        are filled uniformly such that the corresponding norm is 1.

        .. note:: `fill=True` is incompatible with `norm=0` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ParameterError
        If `norm` is not among the valid types defined above

        If `S` is not finite

        If `fill=True` and `norm=0`

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    '''

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `y`:
        `y_frames[i, j] == y[j * hop_length + i]`

    Raises
    ------
    ParameterError
        If `y` is not contiguous in memory, not an `np.ndarray`, or
        not one-dimensional.  See `np.ascontiguous()` for details.

        If `hop_length < 1`, frames cannot advance.

        If `len(y) < frame_length`.

    Examples
    --------
    Extract 2048-sample frames from `y` with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.frame(y, frame_length=2048, hop_length=64)
    array([[ -9.216e-06,   7.710e-06, ...,  -2.117e-06,  -4.362e-07],
           [  2.518e-06,  -6.294e-06, ...,  -1.775e-05,  -6.365e-06],
           ...,
           [ -7.429e-04,   5.173e-03, ...,   1.105e-05,  -5.074e-06],
           [  2.169e-03,   4.867e-03, ...,   3.666e-06,  -5.571e-06]], dtype=float32)

    '''

    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = np.lib.stride_tricks.as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def autocorrelate(y, max_size=None, axis=-1):
    """Bounded auto-correlation

    Parameters
    ----------
    y : np.ndarray
        array to autocorrelate

    max_size  : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to `y.shape[axis]` (unbounded)

    axis : int
        The axis along which to autocorrelate.
        By default, the last axis (-1) is taken.

    Returns
    -------
    z : np.ndarray
        truncated autocorrelation `y*y` along the specified axis.
        If `max_size` is specified, then `z.shape[axis]` is bounded
        to `max_size`.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Compute full autocorrelation of y

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=20, duration=10)
    >>> librosa.autocorrelate(y)
    array([  3.226e+03,   3.217e+03, ...,   8.277e-04,   3.575e-04], dtype=float32)

    Compute onset strength auto-correlation up to 4 seconds

    >>> import matplotlib.pyplot as plt
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    >>> ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
    >>> plt.plot(ac)
    >>> plt.title('Auto-correlation')
    >>> plt.xlabel('Lag (frames)')

    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis))**2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis, overwrite_x=True)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[subslice]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr


def tempo_frequencies(n_bins, hop_length=512, sr=22050):
    '''Compute the frequencies (in beats-per-minute) corresponding
    to an onset auto-correlation or tempogram matrix.

    Parameters
    ----------
    n_bins : int > 0
        The number of lag bins

    hop_length : int > 0
        The number of samples between each bin

    sr : number > 0
        The audio sampling rate

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_bins,)]
        vector of bin frequencies measured in BPM.

        .. note:: `bin_frequencies[0] = +np.inf` corresponds to 0-lag

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.tempo_frequencies(384)
    array([      inf,  2583.984,  1291.992, ...,     6.782,
               6.764,     6.747])
    '''

    bin_frequencies = np.zeros(int(n_bins), dtype=np.float)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


def hpss(S, kernel_size=31, power=2.0, mask=False, margin=1.0):
"""Median-filtering harmonic percussive source separation (HPSS).

If `margin = 1.0`, decomposes an input spectrogram `S = H + P`
where `H` contains the harmonic components,
and `P` contains the percussive components.

If `margin > 1.0`, decomposes an input spectrogram `S = H + P + R`
where `R` contains residual components not included in `H` or `P`.

This implementation is based upon the algorithm described by [1]_ and [2]_.

.. [1] Fitzgerald, Derry.
    "Harmonic/percussive separation using median filtering."
    13th International Conference on Digital Audio Effects (DAFX10),
    Graz, Austria, 2010.

.. [2] Driedger, Müller, Disch.
    "Extending harmonic-percussive separation of audio."
    15th International Society for Music Information Retrieval Conference (ISMIR 2014),
    Taipei, Taiwan, 2014.

Parameters
----------
S : np.ndarray [shape=(d, n)]
    input spectrogram. May be real (magnitude) or complex.

kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
    kernel size(s) for the median filters.

    - If scalar, the same size is used for both harmonic and percussive.
    - If tuple, the first value specifies the width of the
      harmonic filter, and the second value specifies the width
      of the percussive filter.

power : float > 0 [scalar]
    Exponent for the Wiener filter when constructing soft mask matrices.

mask : bool
    Return the masking matrices instead of components.

    Masking matrices contain non-negative real values that
    can be used to measure the assignment of energy from `S`
    into harmonic or percussive components.

    Components can be recovered by multiplying `S * mask_H`
    or `S * mask_P`.


margin : float or tuple (margin_harmonic, margin_percussive)
    margin size(s) for the masks (as described in [2]_)

    - If scalar, the same size is used for both harmonic and percussive.
    - If tuple, the first value specifies the margin of the
      harmonic mask, and the second value specifies the margin
      of the percussive mask.

Returns
-------
harmonic : np.ndarray [shape=(d, n)]
    harmonic component (or mask)

percussive : np.ndarray [shape=(d, n)]
    percussive component (or mask)


See Also
--------
util.softmask

Notes
-----
This function caches at level 30.

Examples
--------
Separate into harmonic and percussive

>>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
>>> D = librosa.stft(y)
>>> H, P = librosa.decompose.hpss(D)

>>> import matplotlib.pyplot as plt
>>> plt.figure()
>>> plt.subplot(3, 1, 1)
>>> librosa.display.specshow(librosa.amplitude_to_db(D,
...                                                  ref=np.max),
...                          y_axis='log')
>>> plt.colorbar(format='%+2.0f dB')
>>> plt.title('Full power spectrogram')
>>> plt.subplot(3, 1, 2)
>>> librosa.display.specshow(librosa.amplitude_to_db(H,
...                                                  ref=np.max),
...                          y_axis='log')
>>> plt.colorbar(format='%+2.0f dB')
>>> plt.title('Harmonic power spectrogram')
>>> plt.subplot(3, 1, 3)
>>> librosa.display.specshow(librosa.amplitude_to_db(P,
...                                                  ref=np.max),
...                          y_axis='log')
>>> plt.colorbar(format='%+2.0f dB')
>>> plt.title('Percussive power spectrogram')
>>> plt.tight_layout()


Or with a narrower horizontal filter

>>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

Just get harmonic/percussive masks, not the spectra

>>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
>>> mask_H
array([[  1.000e+00,   1.469e-01, ...,   2.648e-03,   2.164e-03],
       [  1.000e+00,   2.368e-01, ...,   9.413e-03,   7.703e-03],
       ...,
       [  8.869e-01,   5.673e-02, ...,   4.603e-02,   1.247e-05],
       [  7.068e-01,   2.194e-02, ...,   4.453e-02,   1.205e-05]], dtype=float32)
>>> mask_P
array([[  2.858e-05,   8.531e-01, ...,   9.974e-01,   9.978e-01],
       [  1.586e-05,   7.632e-01, ...,   9.906e-01,   9.923e-01],
       ...,
       [  1.131e-01,   9.433e-01, ...,   9.540e-01,   1.000e+00],
       [  2.932e-01,   9.781e-01, ...,   9.555e-01,   1.000e+00]], dtype=float32)

Separate into harmonic/percussive/residual components by using a margin > 1.0

>>> H, P = librosa.decompose.hpss(D, margin=3.0)
>>> R = D - (H+P)
>>> y_harm = librosa.core.istft(H)
>>> y_perc = librosa.core.istft(P)
>>> y_resi = librosa.core.istft(R)


Get a more isolated percussive component by widening its margin

>>> H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))

"""

if np.iscomplexobj(S):
    S, phase = magphase(S)
else:
    phase = 1

if np.isscalar(kernel_size):
    win_harm = kernel_size
    win_perc = kernel_size
else:
    win_harm = kernel_size[0]
    win_perc = kernel_size[1]

if np.isscalar(margin):
    margin_harm = margin
    margin_perc = margin
else:
    margin_harm = margin[0]
    margin_perc = margin[1]

# margin minimum is 1.0
if margin_harm < 1 or margin_perc < 1:
    raise ParameterError("Margins must be >= 1.0. "
                         "A typical range is between 1 and 10.")

# Compute median filters. Pre-allocation here preserves memory layout.
harm = np.empty_like(S)
harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

perc = np.empty_like(S)
perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

split_zeros = (margin_harm == 1 and margin_perc == 1)

mask_harm = softmask(harm, perc * margin_harm,
                          power=power,
                          split_zeros=split_zeros)

mask_perc = softmask(perc, harm * margin_perc,
                          power=power,
                          split_zeros=split_zeros)

if mask:
    return mask_harm, mask_perc

return ((S * mask_harm) * phase, (S * mask_perc) * phase)


def softmask(X, X_ref, power=1, split_zeros=False):
    '''Robustly compute a softmask operation.

        `M = X**power / (X**power + X_ref**power)`


    Parameters
    ----------
    X : np.ndarray
        The (non-negative) input array corresponding to the positive mask elements

    X_ref : np.ndarray
        The (non-negative) array of reference or background elements.
        Must have the same shape as `X`.

    power : number > 0 or np.inf
        If finite, returns the soft mask computed in a numerically stable way

        If infinite, returns a hard (binary) mask equivalent to `X > X_ref`.
        Note: for hard masks, ties are always broken in favor of `X_ref` (`mask=0`).


    split_zeros : bool
        If `True`, entries where `X` and X`_ref` are both small (close to 0)
        will receive mask values of 0.5.

        Otherwise, the mask is set to 0 for these entries.


    Returns
    -------
    mask : np.ndarray, shape=`X.shape`
        The output mask array

    Raises
    ------
    ParameterError
        If `X` and `X_ref` have different shapes.

        If `X` or `X_ref` are negative anywhere

        If `power <= 0`

    Examples
    --------

    >>> X = 2 * np.ones((3, 3))
    >>> X_ref = np.vander(np.arange(3.0))
    >>> X
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]])
    >>> X_ref
    array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]])
    >>> librosa.util.softmask(X, X_ref, power=1)
    array([[ 1.   ,  1.   ,  0.667],
           [ 0.667,  0.667,  0.667],
           [ 0.333,  0.5  ,  0.667]])
    >>> librosa.util.softmask(X_ref, X, power=1)
    array([[ 0.   ,  0.   ,  0.333],
           [ 0.333,  0.333,  0.333],
           [ 0.667,  0.5  ,  0.333]])
    >>> librosa.util.softmask(X, X_ref, power=2)
    array([[ 1. ,  1. ,  0.8],
           [ 0.8,  0.8,  0.8],
           [ 0.2,  0.5,  0.8]])
    >>> librosa.util.softmask(X, X_ref, power=4)
    array([[ 1.   ,  1.   ,  0.941],
           [ 0.941,  0.941,  0.941],
           [ 0.059,  0.5  ,  0.941]])
    >>> librosa.util.softmask(X, X_ref, power=100)
    array([[  1.000e+00,   1.000e+00,   1.000e+00],
           [  1.000e+00,   1.000e+00,   1.000e+00],
           [  7.889e-31,   5.000e-01,   1.000e+00]])
    >>> librosa.util.softmask(X, X_ref, power=np.inf)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [False, False,  True]], dtype=bool)
    '''
    if X.shape != X_ref.shape:
        raise ParameterError('Shape mismatch: {}!={}'.format(X.shape,
                                                             X_ref.shape))

    if np.any(X < 0) or np.any(X_ref < 0):
        raise ParameterError('X and X_ref must be non-negative')

    if power <= 0:
        raise ParameterError('power must be strictly positive')

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = (Z < np.finfo(dtype).tiny)
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask


def magphase(D, power=1):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.


    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`, raised to `power`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ...,
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ...,
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)


    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ...,
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)

    """

    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase
