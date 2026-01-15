# emg/io/h5_reader.py

from pathlib import Path
import h5py
import numpy as np
from dataclasses import dataclass

EMG_SCALE = 0.195  # 原始值 -> 真实肌电幅值 μV


@dataclass
class EMGData:
    emg: np.ndarray          # (N, C)
    sample_index: np.ndarray # (N,)
    fs: int
    meta: dict


def read_h5_action_only(path: str | Path) -> EMGData:
    """
    Read single h5 file:
    - keep ACTION phase only
    - scale EMG to real amplitude
    - do NOT merge collectors
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        emg = f["emg"][:]                 # (N, C)
        phase = f["phase"][:]             # (N,)
        sample_index = f["sample_index"][:]  # (N,)

        fs = int(f.attrs["fs"])
        meta = dict(f.attrs)

    # ===== Phase decode & ACTION mask =====
    phase_str = np.array([p.decode() for p in phase])
    action_mask = phase_str == "ACTION"

    if not np.any(action_mask):
        raise ValueError(f"No ACTION segment found in {path.name}")

    # ===== Apply mask =====
    emg = emg[action_mask]
    sample_index = sample_index[action_mask]

    # ===== Scale to real EMG amplitude =====
    emg = emg * EMG_SCALE

    # ===== Meta =====
    meta["file"] = path.name
    meta["collector_id"] = meta.get("collector_id", "unknown")

    return EMGData(
        emg=emg,
        sample_index=sample_index,
        fs=fs,
        meta=meta
    )
