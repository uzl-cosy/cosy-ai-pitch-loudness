from __future__ import division
import librosa
import librosa.display
import numpy as np
import scipy
from librosa import util
from librosa import sequence


#----------------------------------------------------------------------------------------------------------------------#
#pYin - f0 Estimation
#----------------------------------------------------------------------------------------------------------------------#


def _cumulative_mean_normalized_difference(
    y_frames, frame_length, win_length, min_period, max_period
):
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    frame_length : int > 0 [scalar]
         length of the frames in samples.

    win_length : int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.

    min_period : int > 0 [scalar]
        minimum period.

    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    # Autocorrelation.
    a = np.fft.rfft(y_frames, frame_length, axis=0)
    b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=0)[win_length:]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms.
    energy_frames = np.cumsum(y_frames ** 2, axis=0)
    energy_frames = energy_frames[win_length:, :] - energy_frames[:-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function.
    yin_frames = energy_frames[0, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[min_period : max_period + 1, :]
    tau_range = np.arange(1, max_period + 1)[:, None]
    cumulative_mean = np.cumsum(yin_frames[1 : max_period + 1, :], axis=0) / tau_range
    yin_denominator = cumulative_mean[min_period - 1 : max_period, :]
    yin_frames = yin_numerator / (yin_denominator + util.tiny(yin_denominator))
    return yin_frames


def _parabolic_interpolation(y_frames):
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=(frame_length, n_frames)]
        position of the parabola optima
    """
    parabolic_shifts = np.zeros_like(y_frames)
    parabola_a = (y_frames[:-2, :] + y_frames[2:, :] - 2 * y_frames[1:-1, :]) / 2
    parabola_b = (y_frames[2:, :] - y_frames[:-2, :]) / 2
    parabolic_shifts[1:-1, :] = -parabola_b / (2 * parabola_a + util.tiny(parabola_a))
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


def pyin(
    y,
    fmin,
    fmax,
    sr=22050,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    n_thresholds=100,
    beta_parameters=(2, 18),
    boltzmann_parameter=2,
    resolution=0.1,
    max_transition_rate=35.92,
    switch_prob=0.01,
    no_trough_prob=0.01,
    fill_na=np.nan,
    center=True,
    pad_mode="reflect",
):
    """Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.

    .. [#] Mauch, Matthias, and Simon Dixon.
        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series.

    fmin: number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.

    fmax: number > 0 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.

    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.

    frame_length : int > 0 [scalar]
         length of the frames in samples.
         By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
         a sampling rate of 22050 Hz.

    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``

    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent pYIN predictions.
        If ``None``, defaults to ``frame_length // 4``.

    n_thresholds : int > 0 [scalar]
        number of thresholds for peak estimation.

    beta_parameters : tuple
        shape parameters for the beta distribution prior over thresholds.

    boltzmann_parameter: number > 0 [scalar]
        shape parameter for the Boltzmann distribution prior over troughs.
        Larger values will assign more mass to smaller periods.

    resolution : float in `(0, 1)`
        Resolution of the pitch bins.
        0.01 corresponds to cents.

    max_transition_rate : float > 0
        maximum pitch transition rate in octaves per second.

    switch_prob : float in ``(0, 1)``
        probability of switching from voiced to unvoiced or vice versa.

    no_trough_prob : float in ``(0, 1)``
        maximum probability to add to global minimum if no trough is below threshold.

    fill_na : None, float, or ``np.nan``
        default value for unvoiced frames of ``f0``.
        If ``None``, the unvoiced frames will contain a best guess value.

    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.

    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="reflect"``),
        ``y`` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(n_frames,)]
        time series of fundamental frequencies in Hertz.

    voiced_flag: np.ndarray [shape=(n_frames,)]
        time series containing boolean flags indicating whether a frame is voiced or not.

    voiced_prob: np.ndarray [shape=(n_frames,)]
        time series containing the probability that a frame is voiced.

    See Also
    --------
    librosa.yin
        Fundamental frequency (F0) estimation using the YIN algorithm.

    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    >>> times = librosa.times_like(f0)


    Overlay F0 over a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    >>> ax.set(title='pYIN fundamental frequency estimation')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    >>> ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    >>> ax.legend(loc='upper right')
    """

    if fmin is None or fmax is None:
        raise librosa.ParameterError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise librosa.ParameterError(
            "win_length={} cannot exceed given frame_length={}".format(
                win_length, frame_length
            )
        )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Check that audio is valid.
    util.valid_audio(y, mono=True)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, frame_length // 2, mode=pad_mode)

    # Frame audio.
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find Yin candidates and probabilities.
    # The implementation here follows the official pYIN software which
    # differs from the method described in the paper.
    # 1. Define the prior over the thresholds.
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    yin_probs = np.zeros_like(yin_frames)
    for i, yin_frame in enumerate(yin_frames.T):
        # 2. For each frame find the troughs.
        is_trough = util.localmin(yin_frame, axis=0)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        # 3. Find the troughs below each threshold.
        trough_heights = yin_frame[trough_index]
        trough_thresholds = trough_heights[:, None] < thresholds[None, 1:]

        # 4. Define the prior over the troughs.
        # Smaller periods are weighted more.
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)
        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions, boltzmann_parameter, n_troughs
        )
        trough_prior[~trough_thresholds] = 0

        # 5. For each threshold add probability to global minimum if no trough is below threshold,
        # else add probability to each trough below threshold biased by prior.
        probs = np.sum(trough_prior * beta_probs, axis=1)
        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    # Refine peak by parabolic interpolation.
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    # Construct transition matrix.
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    # Construct the within voicing transition probabilities
    transition = sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )
    # Include across voicing transition probabilities
    transition = np.block(
        [
            [(1 - switch_prob) * transition, switch_prob * transition],
            [switch_prob * transition, (1 - switch_prob) * transition],
        ]
    )

    # Find pitch bin corresponding to each f0 candidate.
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    # Observation probabilities.
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]
    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob[None, :]) / n_pitch_bins

    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins

    # Viterbi decoding.
    states = sequence.viterbi(observation_probs, transition, p_init=p_init)

    # Find f0 corresponding to each decoded pitch bin.
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    # if fill_na is not None:
    #     f0[~voiced_flag] = fill_na
    return f0, voiced_flag, voiced_prob