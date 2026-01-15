import numpy as np
from typing import Dict

from emg.io.loader import load_emg_dataset
from emg.preprocess.filter import preprocess_emg
from emg.features.extractor import sliding_window_feature_extraction
from emg.features.normalize import SubjectNormalizer
from emg.utils.filename_parser import parse_force_from_filename, parse_gesture_from_filename


def build_dataset(
    root: str,
    window_ms: int = 100,
    step_ms: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Build ML-ready EMG dataset from raw h5 files.

    Returns:
        {
            "X":        (N, C, F),
            "subject":  (N,),
            "collector":(N,),
            "file":     (N,),
            "force":    (N,),   # ⭐ NEW
        }
    """

    datasets = load_emg_dataset(root)

    X_all = []
    subject_all = []
    collector_all = []
    file_all = []
    force_all = []
    gesture_all = []

    subject_id = 0

    for collector_id, emg_list in datasets.items():

        for file_id, emg_data in enumerate(emg_list):

            # ---------- NEW: parse force from filename ----------
            filename = emg_data.meta["filename"]
            force_level = parse_force_from_filename(filename)
            gesture_label = parse_gesture_from_filename(filename)

            # 1) preprocess
            emg_filt = preprocess_emg(emg_data.emg, emg_data.fs)

            # 2) feature extraction
            feats = sliding_window_feature_extraction(
                emg_filt,
                emg_data.fs,
                window_ms=window_ms,
                step_ms=step_ms,
            )  # (n_win, C, F)

            # 3) subject-wise normalization
            normalizer = SubjectNormalizer()
            feats = normalizer.fit_transform(feats)

            n_win = feats.shape[0]

            X_all.append(feats)
            subject_all.append(np.full(n_win, subject_id))
            collector_all.append(np.full(n_win, collector_id))
            file_all.append(np.full(n_win, file_id))
            force_all.append(np.full(n_win, force_level))
            gesture_all.append(np.full(n_win, gesture_label))

            subject_id += 1

    return {
        "X": np.concatenate(X_all, axis=0),
        "subject": np.concatenate(subject_all),
        "collector": np.concatenate(collector_all),
        "file": np.concatenate(file_all),
        "force": np.concatenate(force_all),
        "gesture": np.concatenate(gesture_all),
    }
