import numpy as np

def build_single_force_dataset(dataset, force_level):
    """
    构建单一力水平数据集（A 或 Ck）

    Parameters
    ----------
    dataset : EMGDataset
    force_level : int (10, 20, 30, 40, 50)

    Returns
    -------
    EMGDataset
    """
    mask = dataset.force == force_level
    return dataset.subset(mask)

def build_stratified_cross_force_dataset(dataset, force_levels, samples_per_force):
    """
    构建跨力水平、分层抽样的数据集 B

    Parameters
    ----------
    dataset : EMGDataset
    force_levels : list[int]
        e.g. [10, 20, 30, 40, 50]
    samples_per_force : int
        每个力水平抽取的样本数

    Returns
    -------
    EMGDataset
    """
    indices = []

    for f in force_levels:
        idx_f = np.where(dataset.force == f)[0]
        assert len(idx_f) >= samples_per_force, \
            f"Force {f}% has insufficient samples"

        selected = np.random.choice(
            idx_f,
            size=samples_per_force,
            replace=False
        )
        indices.extend(selected)

    indices = np.array(indices)
    return dataset.subset(indices)

def select_single_device_dataset(dataset, device_name):
    """
    构建单设备数据集

    Parameters
    ----------
    device_name : 'dev1' or 'dev2'
    """
    mask = dataset.device == device_name
    return dataset.subset(mask)

def build_fusion_device_dataset(dataset):
    """
    构建设备 1 + 设备 2 融合的数据集
    特征在 channel 维拼接
    """
    X_fused, y_fused = [], []

    trials = np.unique(dataset.trial)
    wins = np.unique(dataset.win_id)

    for t in trials:
        for w in wins:
            base = (dataset.trial == t) & (dataset.win_id == w)

            d1 = dataset.subset(base & (dataset.device == "dev1"))
            d2 = dataset.subset(base & (dataset.device == "dev2"))

            if len(d1.X) == 1 and len(d2.X) == 1:
                x = np.concatenate([d1.X[0], d2.X[0]], axis=0)
                X_fused.append(x)
                y_fused.append(d1.y[0])

    return dataset.from_arrays(
        X=np.array(X_fused),
        y=np.array(y_fused)
    )

def build_device_specific_dataset(dataset, mode):
    """
    mode:
        'dev1'
        'dev2'
        'fusion'
    """
    if mode in ["dev1", "dev2"]:
        return select_single_device_dataset(dataset, mode)
    elif mode == "fusion":
        return build_fusion_device_dataset(dataset)
    else:
        raise ValueError("Unknown device mode")
