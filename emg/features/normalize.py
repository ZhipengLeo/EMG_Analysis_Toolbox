import numpy as np


class SubjectNormalizer:
    """
    subject-wise, feature-wise z-score normalizer
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        X: (n_windows, n_channels, n_features)
        """
        # 在 window 和 channel 维度上统计
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.std_ = X.std(axis=(0, 1), keepdims=True) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
