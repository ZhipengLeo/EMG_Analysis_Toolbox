import numpy as np
from typing import Dict, List


class EMGDatasetBuilder:
    """
    Step 4: 构建完整 ML 数据集（严格基于已提取的特征）

    假设：
    - trial.features 已经完成：
        - 滑窗
        - 特征提取
        - subject-wise, feature-wise z-score
    """

    def __init__(self):
        pass

    @staticmethod
    def _process_single_trial(
        features: np.ndarray,
        subject_id: int,
        force_id: int,
        trial_id: int,
        gesture_label: int
    ) -> Dict[str, np.ndarray]:
        """
        单个 trial → 数据集样本

        features: (n_windows, 64, 8)
        """

        n_windows = features.shape[0]

        return {
            "X": features,
            "gesture": np.full(n_windows, gesture_label, dtype=int),
            "force": np.full(n_windows, force_id, dtype=int),
            "subject": np.full(n_windows, subject_id, dtype=int),
            "trial": np.full(n_windows, trial_id, dtype=int),
        }

    def build(self, subjects: List) -> Dict[str, np.ndarray]:
        """
        subjects:
            subject.forces[force_id].trials[trial_id].features

        返回完整 Dataset（不划分）
        """

        X_all = []
        gesture_all = []
        force_all = []
        subject_all = []
        trial_all = []

        for subject_id, subject in enumerate(subjects):
            for force_id, force_block in enumerate(subject.forces):
                for trial_id, trial in enumerate(force_block.trials):

                    assert hasattr(trial, "features"), \
                        "trial.features 不存在，请先完成 Step 3"

                    out = self._process_single_trial(
                        features=trial.features,   # (n_windows, 64, 8)
                        subject_id=subject_id,
                        force_id=force_id,
                        trial_id=trial_id,
                        gesture_label=trial.gesture
                    )

                    X_all.append(out["X"])
                    gesture_all.append(out["gesture"])
                    force_all.append(out["force"])
                    subject_all.append(out["subject"])
                    trial_all.append(out["trial"])

        dataset = {
            "X": np.concatenate(X_all, axis=0),
            "gesture": np.concatenate(gesture_all),
            "force": np.concatenate(force_all),
            "subject": np.concatenate(subject_all),
            "trial": np.concatenate(trial_all),
        }

        return dataset
