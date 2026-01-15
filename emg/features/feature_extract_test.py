import numpy as np

from emg.features.extractor import sliding_window_feature_extraction
from emg.features.normalize import SubjectNormalizer


def generate_fake_emg(
    n_samples: int,
    fs: int,
    force_level: float = 1.0,
    noise_std: float = 0.02
):
    """
    生成一段类 EMG 信号（单通道）
    force_level: 控制幅值大小，用来模拟不同力水平
    """
    t = np.arange(n_samples) / fs

    # 模拟：带限噪声 + 幅值调制
    emg = force_level * (
        0.3 * np.sin(2 * np.pi * 60 * t) +
        0.2 * np.sin(2 * np.pi * 120 * t) +
        noise_std * np.random.randn(n_samples)
    )

    return emg


if __name__ == "__main__":

    # =========================
    # 1. 基本参数
    # =========================
    fs = 2000                # Hz
    duration = 2.0           # 秒
    n_samples = int(fs * duration)

    # =========================
    # 2. 生成单通道 EMG
    # =========================
    emg_1ch = generate_fake_emg(
        n_samples=n_samples,
        fs=fs,
        force_level=1.0
    )

    # sliding_window_feature_extraction 期望 (N, C)
    filtered_emg = emg_1ch[:, None]   # shape: (N, 1)

    print("Input EMG shape:", filtered_emg.shape)

    # =========================
    # 3. 特征提取
    # =========================
    X = sliding_window_feature_extraction(
        filtered_emg,
        fs=fs,
        window_ms=100,
        step_ms=50
    )

    print("Raw feature shape:", X.shape)
    # (n_windows, 1, 8)

    # =========================
    # 4. subject-wise 归一化
    # =========================
    normalizer = SubjectNormalizer()
    X_norm = normalizer.fit_transform(X)

    print("Normalized feature shape:", X_norm.shape)

    # =========================
    # 5. 简单 sanity check
    # =========================
    mean_per_feature = X_norm.mean(axis=(0, 1))
    std_per_feature = X_norm.std(axis=(0, 1))

    print("\nFeature-wise mean after normalization:")
    print(mean_per_feature)

    print("\nFeature-wise std after normalization:")
    print(std_per_feature)
