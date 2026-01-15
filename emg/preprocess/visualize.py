# emg/preprocess/visualize.py

import numpy as np
import matplotlib.pyplot as plt


def plot_emg_before_after(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: int,
    channels=(0, 1),
    title: str = "",
):
    """
    Plot raw vs filtered EMG for selected channels.
    """
    n = raw.shape[0]
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 4))

    for ch in channels:
        plt.plot(t, raw[:, ch])
        plt.plot(t, filtered[:, ch])

    plt.xlabel("Time (s)")
    plt.ylabel("EMG Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()
