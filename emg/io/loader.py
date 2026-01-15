# emg/io/loader.py

from pathlib import Path
from collections import defaultdict
from .h5_reader import read_h5_action_only, EMGData


def load_emg_dataset(root: str | Path) -> dict[str, list[EMGData]]:
    """
    Recursively load all h5 files.

    Returns:
        {
            "dev0": [EMGData, EMGData, ...],
            "dev1": [EMGData, EMGData, ...]
        }
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    datasets = defaultdict(list)

    for h5_file in sorted(root.rglob("*.h5")):
        try:
            emg_data = read_h5_action_only(h5_file)
            cid = emg_data.meta["collector_id"]
            datasets[cid].append(emg_data)
        except Exception as e:
            print(f"[SKIP] {h5_file.name}: {e}")

    return dict(datasets)
