import numpy as np
from typing import Dict, Tuple


def split_by_force_level(
        dataset: Dict[str, np.ndarray],
        test_force: int,
        train_forces_mixed: Tuple[int, ...],
        n_test_trials: int = 2,
        n_train_trials_per_force: int = 2,
        seed: int = 0,
        balanced_split: bool = False,
        warn_on_window_mismatch: bool = True,   # 👈 新增：是否打印 warning
):
    """
    Trial-balanced force-level split.

    设计原则：
        - test / train 在 trial 级别严格不重叠
        - mixed-force balanced：trial 数平衡，不强制 window 数平衡
        - window 数不一致是现实数据现象，不视为错误
    """

    force = dataset["force"]
    trial = dataset["trial"]
    gesture = dataset["gesture"]

    rng = np.random.default_rng(seed)

    forces = np.unique(force)
    gestures = np.unique(gesture)

    assert test_force in forces
    assert test_force not in train_forces_mixed, \
        "train_forces_mixed must not include test_force"

    # ============================================================
    # 1️⃣ Test set
    # ============================================================
    test_indices = []

    for g in gestures:
        idx = np.where((force == test_force) & (gesture == g))[0]
        if len(idx) == 0:
            continue

        trials = np.unique(trial[idx])
        assert len(trials) >= n_test_trials, \
            f"Not enough trials for force={test_force}, gesture={g}"

        rng.shuffle(trials)
        selected_trials = trials[:n_test_trials]

        for t in selected_trials:
            test_indices.extend(idx[trial[idx] == t])

    test_indices = np.array(test_indices, dtype=np.int64)
    mask_test = np.zeros(len(force), dtype=bool)
    mask_test[test_indices] = True

    # ============================================================
    # 2️⃣ Same-force training
    # ============================================================
    mask_train_same = (force == test_force) & (~mask_test)

    # ============================================================
    # 3️⃣ Cross-force training
    # ============================================================
    mask_train_cross = np.isin(force, train_forces_mixed)

    # ============================================================
    # 4️⃣ Mixed-force full
    # ============================================================
    mask_train_mixed_full = (~mask_test)

    # ============================================================
    # 5️⃣ Mixed-force balanced (trial-balanced)
    # ============================================================
    balanced_indices = []

    if balanced_split:
        for f in forces:      # 包含 test_force
            for g in gestures:
                idx = np.where(
                    (force == f) &
                    (gesture == g) &
                    (~mask_test)
                )[0]

                if len(idx) == 0:
                    continue

                trials = np.unique(trial[idx])
                assert len(trials) >= n_train_trials_per_force, \
                    f"Not enough trials for force={f}, gesture={g}"

                rng.shuffle(trials)
                selected_trials = trials[:n_train_trials_per_force]

                for t in selected_trials:
                    balanced_indices.extend(idx[trial[idx] == t])

    balanced_indices = np.array(balanced_indices, dtype=np.int64)
    mask_train_mixed_balanced = np.zeros(len(force), dtype=bool)
    mask_train_mixed_balanced[balanced_indices] = True

    # ============================================================
    # 6️⃣ Safety checks（trial-level）
    # ============================================================
    assert not np.any(mask_test & mask_train_same)
    assert not np.any(mask_test & mask_train_cross)
    assert not np.any(mask_test & mask_train_mixed_full)
    assert not np.any(mask_test & mask_train_mixed_balanced)

    # window 数不一致 → warning（不是错误）
    if balanced_split and warn_on_window_mismatch:
        for g in gestures:
            n_same = np.sum(mask_train_same & (gesture == g))
            n_mixed = np.sum(mask_train_mixed_balanced & (gesture == g))

            if n_same != n_mixed:
                print(
                    f"[WARN] Gesture {g}: "
                    f"same-force windows={n_same}, "
                    f"mixed-force windows={n_mixed} "
                    "(trial-balanced, window-unbalanced)"
                )

    # ============================================================
    # 7️⃣ Return
    # ============================================================
    return {
        "train_same": _subset(dataset, mask_train_same),
        "train_cross": _subset(dataset, mask_train_cross),
        "train_mixed_full": _subset(dataset, mask_train_mixed_full),
        "train_mixed_balanced": _subset(dataset, mask_train_mixed_balanced),
        "test": _subset(dataset, mask_test),
    }


def _subset(
        dataset: Dict[str, np.ndarray],
        mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Sample-level slicing while keeping feature alignment.
    """
    subset = {}

    for k, v in dataset.items():
        if k == "single_feature_data":
            subset[k] = {
                feat_name: feat_data[mask]
                for feat_name, feat_data in v.items()
            }
        else:
            subset[k] = v[mask]

    return subset
