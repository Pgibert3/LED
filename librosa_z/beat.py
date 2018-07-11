import numpy as np
import scipy
from . import exceptions
from . import util

def beat_track(y=None, sr=22050, onset_envelope=None, hop_length=512,
               start_bpm=120.0, tightness=100, trim=True, bpm=None,
               units='frames'):
    r'''Dynamic programming beat tracker.

    Beats are detected in three stages, following the method of [1]_:
      1. Measure onset strength
      2. Estimate tempo from onset correlation
      3. Pick peaks in onset strength approximately consistent with estimated
         tempo

    .. [1] Ellis, Daniel PW. "Beat tracking by dynamic programming."
           Journal of New Music Research 36.1 (2007): 51-60.
           http://labrosa.ee.columbia.edu/projects/beattrack/


    Parameters
    ----------

    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope : np.ndarray [shape=(n,)] or None
        (optional) pre-computed onset strength envelope.

    hop_length : int > 0 [scalar]
        number of audio samples between successive `onset_envelope` values

    start_bpm  : float > 0 [scalar]
        initial guess for the tempo estimator (in beats per minute)

    tightness  : float [scalar]
        tightness of beat distribution around tempo

    trim       : bool [scalar]
        trim leading/trailing beats with weak onsets

    bpm        : float [scalar]
        (optional) If provided, use `bpm` as the tempo instead of
        estimating it from `onsets`.

    units : {'frames', 'samples', 'time'}
        The units to encode detected beat events in.
        By default, 'frames' are used.


    Returns
    -------

    tempo : float [scalar, non-negative]
        estimated global tempo (in beats per minute)

    beats : np.ndarray [shape=(m,)]
        estimated beat event locations in the specified units
        (default is frame indices)

    .. note::
        If no onset strength could be detected, beat_tracker estimates 0 BPM
        and returns an empty list.


    Raises
    ------
    ParameterError
        if neither `y` nor `onset_envelope` are provided

        or if `units` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    librosa.onset.onset_strength


    Examples
    --------
    Track beats using time series input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())

    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> tempo
    64.599609375


    Print the first 20 beat frames

    >>> beats[:20]
    array([ 320,  357,  397,  436,  480,  525,  569,  609,  658,
            698,  737,  777,  817,  857,  896,  936,  976, 1016,
           1055, 1095])


    Or print them as timestamps

    >>> librosa.frames_to_time(beats[:20], sr=sr)
    array([  7.43 ,   8.29 ,   9.218,  10.124,  11.146,  12.19 ,
            13.212,  14.141,  15.279,  16.208,  17.113,  18.042,
            18.971,  19.9  ,  20.805,  21.734,  22.663,  23.591,
            24.497,  25.426])


    Track beats using a pre-computed onset envelope

    >>> onset_env = librosa.onset.onset_strength(y, sr=sr,
    ...                                          aggregate=np.median)
    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
    ...                                        sr=sr)
    >>> tempo
    64.599609375
    >>> beats[:20]
    array([ 320,  357,  397,  436,  480,  525,  569,  609,  658,
            698,  737,  777,  817,  857,  896,  936,  976, 1016,
           1055, 1095])


    Plot the beat events against the onset strength envelope

    >>> import matplotlib.pyplot as plt
    >>> hop_length = 512
    >>> plt.figure(figsize=(8, 4))
    >>> times = librosa.frames_to_time(np.arange(len(onset_env)),
    ...                                sr=sr, hop_length=hop_length)
    >>> plt.plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> plt.vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> plt.legend(frameon=True, framealpha=0.75)
    >>> # Limit the plot to a 15-second window
    >>> plt.xlim(15, 30)
    >>> plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> plt.tight_layout()
    '''

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
            raise ParameterError('onset_envelope must be provided')

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return (0, np.array([], dtype=int))

    # Estimate BPM if one was not provided
    if bpm is None:
        bpm = tempo(onset_envelope=onset_envelope,
                    sr=sr,
                    hop_length=hop_length,
                    start_bpm=start_bpm)[0]

    # Then, run the tracker
    beats = __beat_tracker(onset_envelope,
                           bpm,
                           float(sr) / hop_length,
                           tightness,
                           trim)

    if units == 'frames':
        pass
    elif units == 'samples':
        beats = util.frames_to_samples(beats, hop_length=hop_length)
    elif units == 'time':
        beats = util.frames_to_time(beats, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError('Invalid unit type: {}'.format(units))

    return (bpm, beats)


@cache(level=30)
def tempo(y=None, sr=22050, onset_envelope=None, hop_length=512, start_bpm=120,
          std_bpm=1.0, ac_size=8.0, max_tempo=320.0, aggregate=np.mean):
    """Estimate the tempo (beats per minute)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of the time series

    onset_envelope    : np.ndarray [shape=(n,)]
        pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length of the time series

    start_bpm : float [scalar]
        initial guess of the BPM

    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution

    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window

    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold

    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.

    Returns
    -------
    tempo : np.ndarray [scalar]
        estimated tempo (beats per minute)

    See Also
    --------
    librosa.onset.onset_strength
    librosa.feature.tempogram

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Estimate a static tempo
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> onset_env = librosa.onset.onset_strength(y, sr=sr)
    >>> tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    >>> tempo
    array([129.199])

    >>> # Or a dynamic tempo
    >>> dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
    ...                             aggregate=None)
    >>> dtempo
    array([ 143.555,  143.555,  143.555, ...,  161.499,  161.499,
            172.266])


    Plot the estimated tempo against the onset autocorrelation

    >>> import matplotlib.pyplot as plt
    >>> # Convert to scalar
    >>> tempo = np.asscalar(tempo)
    >>> # Compute 2-second windowed autocorrelation
    >>> hop_length = 512
    >>> ac = librosa.autocorrelate(onset_env, 2 * sr // hop_length)
    >>> freqs = librosa.tempo_frequencies(len(ac), sr=sr,
    ...                                   hop_length=hop_length)
    >>> # Plot on a BPM axis.  We skip the first (0-lag) bin.
    >>> plt.figure(figsize=(8,4))
    >>> plt.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
    ...              label='Onset autocorrelation', basex=2)
    >>> plt.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--',
    ...            label='Tempo: {:.2f} BPM'.format(tempo))
    >>> plt.xlabel('Tempo (BPM)')
    >>> plt.grid()
    >>> plt.title('Static tempo estimation')
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')

    Plot dynamic tempo estimates over a tempogram

    >>> plt.figure()
    >>> tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
    ...                                hop_length=hop_length)
    >>> librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
    >>> plt.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo,
    ...          color='w', linewidth=1.5, label='Tempo estimate')
    >>> plt.title('Dynamic tempo estimation')
    >>> plt.legend(frameon=True, framealpha=0.75)
    """

    if start_bpm <= 0:
        raise ParameterError('start_bpm must be strictly positive')

    win_length = np.asscalar(core.time_to_frames(ac_size, sr=sr,
                                                 hop_length=hop_length))

    tg = tempogram(y=y, sr=sr,
                   onset_envelope=onset_envelope,
                   hop_length=hop_length,
                   win_length=win_length)

    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:
        tg = aggregate(tg, axis=1, keepdims=True)

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = util.tempo_frequencies(tg.shape[0], hop_length=hop_length, sr=sr)

    # Weight the autocorrelation by a log-normal distribution
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm)**2)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = np.argmax(bpms < max_tempo)
        prior[:max_idx] = 0

    # Really, instead of multiplying by the prior, we should set up a
    # probabilistic model for tempo and add log-probabilities.
    # This would give us a chance to recover from null signals and
    # rely on the prior.
    # it would also make time aggregation much more natural

    # Get the maximum, weighted by the prior
    best_period = np.argmax(tg * prior[:, np.newaxis], axis=0)

    tempi = bpms[best_period]
    # Wherever the best tempo is index 0, return start_bpm
    tempi[best_period == 0] = start_bpm
    return tempi


def __beat_tracker(onset_envelope, bpm, fft_res, tightness, trim):
    """Internal function that tracks beats in an onset strength envelope.

    Parameters
    ----------
    onset_envelope : np.ndarray [shape=(n,)]
        onset strength envelope

    bpm : float [scalar]
        tempo estimate

    fft_res  : float [scalar]
        resolution of the fft (sr / hop_length)

    tightness: float [scalar]
        how closely do we adhere to bpm?

    trim : bool [scalar]
        trim leading/trailing beats with weak onsets?

    Returns
    -------
    beats : np.ndarray [shape=(n,)]
        frame numbers of beat events
    """

    if bpm <= 0:
        raise ParameterError('bpm must be strictly positive')

    # convert bpm to a sample period for searching
    period = round(60.0 * fft_res / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore = __beat_local_score(onset_envelope, period)

    # run the DP
    backlink, cumscore = __beat_track_dp(localscore, period, tightness)

    # get the position of the last beat
    beats = [__last_beat(cumscore)]

    # Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    # Put the beats in ascending order
    # Convert into an array of frame numbers
    beats = np.array(beats[::-1], dtype=int)

    # Discard spurious trailing beats
    beats = __trim_beats(localscore, beats, trim)

    return beats


# -- Helper functions for beat tracking
def __normalize_onsets(onsets):
    '''Maps onset strength function into the range [0, 1]'''

    norm = onsets.std(ddof=1)
    if norm > 0:
        onsets = onsets / norm
    return onsets


def __beat_local_score(onset_envelope, period):
    '''Construct the local score for an onset envlope and given period'''

    window = np.exp(-0.5 * (np.arange(-period, period+1)*32.0/period)**2)
    return scipy.signal.convolve(__normalize_onsets(onset_envelope),
                                 window,
                                 'same')


def __beat_track_dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""

    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed
    if tightness <= 0:
        raise ParameterError('tightness must be strictly positive')

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(- window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore


def __last_beat(cumscore):
    """Get the last beat from the cumulative score array"""

    maxes = util.localmax(cumscore)
    med_score = np.median(cumscore[np.argwhere(maxes)])

    # The last of these is the last beat (since score generally increases)
    return np.argwhere((cumscore * maxes * 2 > med_score)).max()


def __trim_beats(localscore, beats, trim):
    """Final post-processing: throw out spurious leading/trailing beats"""

    smooth_boe = scipy.signal.convolve(localscore[beats],
                                       scipy.signal.hann(5),
                                       'same')

    if trim:
        threshold = 0.5 * ((smooth_boe**2).mean()**0.5)
    else:
        threshold = 0.0

    valid = np.argwhere(smooth_boe > threshold)

    return beats[valid.min():valid.max()]


def tempogram(y=None, sr=22050, onset_envelope=None, hop_length=512,
              win_length=384, center=True, window='hann', norm=np.inf):
    '''Compute the tempogram: local autocorrelation of the onset strength envelope. [1]_

    .. [1] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        Audio time series.

    sr : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope : np.ndarray [shape=(n,) or (m, n)] or None
        Optional pre-computed onset strength envelope as provided by
        `onset.onset_strength`.

        If multi-dimensional, tempograms are computed independently for each
        band (first dimension).

    hop_length : int > 0
        number of audio samples between successive onset measurements

    win_length : int > 0
        length of the onset autocorrelation window (in frames/onset measurements)
        The default settings (384) corresponds to `384 * hop_length / sr ~= 8.9s`.

    center : bool
        If `True`, onset autocorrelation windows are centered.
        If `False`, windows are left-aligned.

    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `core.stft`.

    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode.  Set to `None` to disable normalization.

    Returns
    -------
    tempogram : np.ndarray [shape=(win_length, n) or (m, win_length, n)]
        Localized autocorrelation of the onset strength envelope.

        If given multi-band input (`onset_envelope.shape==(m,n)`) then
        `tempogram[i]` is the tempogram of `onset_envelope[i]`.

    Raises
    ------
    ParameterError
        if neither `y` nor `onset_envelope` are provided

        if `win_length < 1`

    See Also
    --------
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.core.stft


    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                       hop_length=hop_length)
    >>> # Compute global onset autocorrelation
    >>> ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    >>> ac_global = librosa.util.normalize(ac_global)
    >>> # Estimate the global tempo for display purposes
    >>> tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
    ...                            hop_length=hop_length)[0]

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 8))
    >>> plt.subplot(4, 1, 1)
    >>> plt.plot(oenv, label='Onset strength')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.subplot(4, 1, 2)
    >>> # We'll truncate the display to a narrower range of tempi
    >>> librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo')
    >>> plt.axhline(tempo, color='w', linestyle='--', alpha=1,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> plt.legend(frameon=True, framealpha=0.75)
    >>> plt.subplot(4, 1, 3)
    >>> x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
    ...                 num=tempogram.shape[0])
    >>> plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    >>> plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    >>> plt.xlabel('Lag (seconds)')
    >>> plt.axis('tight')
    >>> plt.legend(frameon=True)
    >>> plt.subplot(4,1,4)
    >>> # We can also plot on a BPM axis
    >>> freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    >>> plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
    ...              label='Mean local autocorrelation', basex=2)
    >>> plt.semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
    ...              label='Global autocorrelation', basex=2)
    >>> plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> plt.legend(frameon=True)
    >>> plt.xlabel('BPM')
    >>> plt.axis('tight')
    >>> plt.grid()
    >>> plt.tight_layout()
    '''

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError('win_length must be a positive integer')

    ac_window = get_window(window, win_length, fftbins=True)

    if onset_envelope is None:
            raise ParameterError('onset_envelope must be provided')

    else:
        # Force row-contiguity to avoid framing errors below
        onset_envelope = np.ascontiguousarray(onset_envelope)

    if onset_envelope.ndim > 1:
        # If we have multi-band input, iterate over rows
        return np.asarray([tempogram(onset_envelope=oe_subband,
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     center=center,
                                     window=window,
                                     norm=norm) for oe_subband in onset_envelope])

    # Center the autocorrelation windows
    n = len(onset_envelope)

    if center:
        onset_envelope = np.pad(onset_envelope, int(win_length // 2),
                                mode='linear_ramp', end_values=[0, 0])

    # Carve onset envelope into frames
    odf_frame = util.frame(onset_envelope,
                           frame_length=win_length,
                           hop_length=1)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[:, :n]

    # Window, autocorrelate, and normalize
    return util.normalize(autocorrelate(odf_frame * ac_window[:, np.newaxis],
                                        axis=0),
                          norm=norm, axis=0)
