import numpy as np
from typing import Dict

from emg.io.loader import load_emg_dataset
from emg.preprocess.filter import preprocess_emg
from emg.features.extractor import sliding_window_feature_extraction, build_single_feature_datasets
from emg.utils.filename_parser import (
    parse_force_from_filename,
    parse_gesture_from_filename,
)



def build_dataset(
    root: str,
    window_ms: int = 100,
    step_ms: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Build ML-ready EMG dataset (trial-based).

    Each file == one trial.
    Normalization is NOT performed here.

    Returns:
        {
            "X":         (N, C, F),
            "collector": (N,),
            "force":     (N,),
            "gesture":   (N,),
            "trial":     (N,),
        }
    """

    datasets = load_emg_dataset(root)

    X_all = []
    collector_all = []
    force_all = []
    gesture_all = []
    trial_all = []

    trial_id = 0  # global unique trial id

    # 存储单特征数据
    single_feature_data_all = []

    for collector_id, emg_list in datasets.items():

        for emg_data in emg_list:

            filename = emg_data.meta["filename"]

            force_level = parse_force_from_filename(filename)
            gesture_label = parse_gesture_from_filename(filename) - 1  # 0-based

            # 1️⃣ preprocess
            emg_filt = preprocess_emg(emg_data.emg, emg_data.fs)

            # 2️⃣ sliding-window feature extraction
            feats = sliding_window_feature_extraction(
                emg_filt,
                emg_data.fs,
                window_ms=window_ms,
                step_ms=step_ms,
            )  # (n_win, C, F)

            n_win = feats.shape[0]

            # 3️⃣ store (each file == one trial)
            X_all.append(feats)
            collector_all.append(np.full(n_win, collector_id))
            force_all.append(np.full(n_win, force_level))
            gesture_all.append(np.full(n_win, gesture_label))
            trial_all.append(np.full(n_win, trial_id))

            # 4️⃣ Convert to single-feature datasets
            single_feature_data = build_single_feature_datasets(
                feats,
                keep_feature_dim=True   # 如果需要保留特征维度
            )

            single_feature_data_all.append(single_feature_data)

            trial_id += 1

    # 将所有单特征数据拼接起来
    single_feature_data_concatenated = {name: np.concatenate([data[name] for data in single_feature_data_all], axis=0)
                                        for name in single_feature_data_all[0].keys()}

    return {
        "X": np.concatenate(X_all, axis=0),
        "collector": np.concatenate(collector_all),
        "force": np.concatenate(force_all),
        "gesture": np.concatenate(gesture_all),
        "trial": np.concatenate(trial_all),
        # 返回单特征数据字典
        "single_feature_data": single_feature_data_concatenated
    }

