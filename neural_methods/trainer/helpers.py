"""
-------------------------------------------------------------------------------
Created: 05.06.2024, 20:41
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
-------------------------------------------------------------------------------
Purpose: To enable the easy access of some helper functions necessary for evaluation during training.
-------------------------------------------------------------------------------
"""
# imports
import numpy as np
from scipy import signal


# NOTE: this function is taken from the ippg-toolbox https://github.com/BeCuriousS/ippg-toolbox
def compute_hr_from_spectrum_peak_det(seq,
                                      sample_freq,
                                      freq_limits_bpm=(30, 200),
                                      freq_res_bpm=0.5,
                                      fft_window='hamming',
                                      peak_prom=0.5,
                                      verbose=False):
    """Compute the heart rate from the spectrum using a peak detection. This is useful if the spectrum probably exhibits higher peaks for the harmonic frequencies and a simple maximum detection would fail.

    Parameters
    ----------
    seq : 1D array like object
        any continuous physiological sequence containing the heart beat
    sample_freq : float
        the sampling frequency of the input sequence
    freq_limits_bpm : tuple, optional
        the boundaries within to search for the heart rate in beats per minute, by default (30, 200)
    freq_res_bpm : float, optional
        the resolution of the frequency for the power spectral density computation, by default 0.5
    fft_window : str, optional
        the window to use when computing the power spectral density, by default 'hamming'
    peak_prom : float, optional
        the peak prominence to use to identify the peaks whereas the minimum frequency is then assumed to represent the heart rate, by default 0.5
    verbose : bool, optional
        if additional results from the computation should also be returned, by default False

    Returns
    -------
    float
        the computed heart rate; if no peak could be found returns np.nan
    dict
        dict containing {'freq', 'power', 'power_n', 'peaks', 'nfft'}; is only returned, when verbose is True
    """
    seq = cstm_squeeze(seq)
    freq_limits_hz = (freq_limits_bpm[0]/60, freq_limits_bpm[1]/60)
    nyquist = sample_freq/2
    n = (60*2*nyquist)/freq_res_bpm
    # filter the signal to allow peak prominence to work correctly
    seq = apply_filter(seq, sample_freq, 3, freq_limits_bpm)
    freq, power = signal.periodogram(
        seq, fs=sample_freq, window=fft_window, nfft=n)
    # normalize to apply prominence properly
    power_n = power/power.max()
    peaks, _ = signal.find_peaks(power_n, prominence=peak_prom)
    # check if at least one peak could be found
    if len(peaks):
        hr = freq[peaks[0]] * 60
    else:
        hr = np.nan

    if verbose:
        return hr, {'freq': freq,
                    'power': power,
                    'power_n': power_n,
                    'peaks': peaks,
                    'nfft': n}
    return hr


def apply_filter(seq,
                 sample_freq,
                 order,
                 cutoff_bpm,
                 axis=0):
    """Filter some signal using a butterworth lowpass or bandpass filter. Note that the resulting filter order is 4 * order because: order * 2 for bandpass butterworth and order * 2 for zero-phase filtering.

    Parameters
    ----------
    seq : 1D array like object
        the sequence to be filtered
    sample_freq : float
        the sampling frequency of the sequence in Hz
    order : int
        the filter order. Note that the resulting filter order will be 4*order
    cutoff_bpm : int or tuple of integers (e.g. (30, 200))
        the cutoff frequencies for the filter in beats per minute not Hz! (for convenience); if int than a lowpass filter is applied; if tuple than a bandpass filter is applied
    axis : int, optional. The axis of the sequence along which the filtering is applied, by default 0. Should be applied for the time axis.

    Returns
    -------
    numpy.ndarray
        the filtered input sequence
    """
    seq = cstm_squeeze(seq)
    if type(cutoff_bpm) == tuple:
        cutoff_hz = (cutoff_bpm[0]/60, cutoff_bpm[1]/60)
        btype = 'bandpass'
    else:
        cutoff_hz = cutoff_bpm/60
        btype = 'lowpass'
    sos = signal.butter(order, cutoff_hz, btype=btype,
                        fs=sample_freq, output='sos')
    seq_filt = signal.sosfiltfilt(sos, seq, axis=axis)
    return seq_filt


def cstm_squeeze(seq):
    if type(seq) == list:
        seq = np.asarray(seq)
    if type(seq) == np.ndarray and len(seq.shape) > 1 and seq.size > 1:
        seq = np.squeeze(seq)
    return seq
