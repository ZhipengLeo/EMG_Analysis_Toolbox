import numpy as np
from scipy.signal import welch


def power_spectrum(x: np.ndarray, fs: int):
    f, pxx = welch(x, fs=fs, nperseg=len(x))
    return f, pxx


def ttp(x: np.ndarray, fs: int) -> float:
    _, pxx = power_spectrum(x, fs)
    return np.sum(pxx)


def mf(x: np.ndarray, fs: int) -> float:
    f, pxx = power_spectrum(x, fs)
    cumulative = np.cumsum(pxx)
    return f[np.where(cumulative >= cumulative[-1] / 2)[0][0]]


def mpf(x: np.ndarray, fs: int) -> float:
    f, pxx = power_spectrum(x, fs)
    return np.sum(f * pxx) / np.sum(pxx)
