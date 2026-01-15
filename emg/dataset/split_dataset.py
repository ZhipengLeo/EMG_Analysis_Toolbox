import numpy as np
from typing import Dict, Tuple


def split_by_force_level(
    dataset: Dict[str, np.ndarray],
    test_force: int,
    train_forces_mixed: Tuple[int, ...],
):
    """
    Split dataset according to force level experiment design.

    Args:
        dataset: output of build_dataset()
        test_force: force level used for test set B
        train_forces_mixed: force levels used to build training set C

    Returns:
        {
            "train_A": {...},   # same force as test
            "train_C": {...},   # mixed forces
            "test_B":  {...},   # fixed force
        }
    """

    force = dataset["force"]

    # ---------- masks ----------
    mask_test_B = force == test_force
    mask_train_A = force == test_force
    mask_train_C = np.isin(force, train_forces_mixed)

    train_A = _subset_dataset(dataset, mask_train_A)
    train_C = _subset_dataset(dataset, mask_train_C)
    test_B = _subset_dataset(dataset, mask_test_B)

    return {
        "train_A": train_A,
        "train_C": train_C,
        "test_B": test_B,
    }


def _subset_dataset(
    dataset: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Apply boolean mask to all fields in dataset.
    """
    return {k: v[mask] for k, v in dataset.items()}
