# emg/preprocess/filter.py

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    emg: np.ndarray,
    fs: int,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """
    Band-pass filter for multi-channel EMG.

    Args:
        emg: (N, C)
        fs: sampling frequency
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, emg, axis=0)


def notch_filter(
    emg: np.ndarray,
    fs: int,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Notch filter at given frequency (e.g. 50 Hz).
    """
    nyq = 0.5 * fs
    w0 = freq / nyq

    b, a = iirnotch(w0, q)
    return filtfilt(b, a, emg, axis=0)


def preprocess_emg(
    emg: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Standard EMG preprocessing:
    1) Band-pass 20–500 Hz
    2) 50 Hz notch
    """
    emg_filt = bandpass_filter(emg, fs)
    emg_filt = notch_filter(emg_filt, fs)
    return emg_filt
