import numpy as np

from .time_domain import rms, mav, var, ssc, aac
from .freq_domain import ttp, mf, mpf


FEATURE_NAMES = [
    "RMS", "MAV", "VAR", "SSC", "AAC", "TTP", "MF", "MPF"
]


def extract_features_from_window(
    window: np.ndarray,
    fs: int
) -> np.ndarray:
    """
    window: (window_len,)
    return: (8,)
    """
    return np.array([
        rms(window),
        mav(window),
        var(window),
        ssc(window),
        aac(window),
        ttp(window, fs),
        mf(window, fs),
        mpf(window, fs),
    ])


def sliding_window_feature_extraction(
    emg: np.ndarray,
    fs: int,
    window_ms: int = 100,
    step_ms: int = 50
) -> np.ndarray:
    """
    emg: (n_samples, n_channels)
    return: (n_windows, n_channels, n_features)
    """
    win_len = int(fs * window_ms / 1000)
    step = int(fs * step_ms / 1000)

    n_samples, n_channels = emg.shape
    features = []

    for start in range(0, n_samples - win_len + 1, step):
        end = start + win_len
        win_feat = []

        for ch in range(n_channels):
            f = extract_features_from_window(emg[start:end, ch], fs)
            win_feat.append(f)

        features.append(win_feat)

    return np.array(features)
