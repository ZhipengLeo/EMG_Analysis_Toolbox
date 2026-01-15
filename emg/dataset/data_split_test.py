from emg.dataset.build_dataset import build_dataset
from emg.dataset.split_dataset import split_by_force_level


def main():
    dataset = build_dataset(
        root="F:\跨力手势分类实验20260113\EMG_Data",
        window_ms=100,
        step_ms=50,
    )

    splits = split_by_force_level(
        dataset,
        test_force=30,              # 例如：30% MVC
        train_forces_mixed=(10, 50) # 混合 10% + 50%
    )

    for name, ds in splits.items():
        print(f"\n{name}")
        print("X:", ds["X"].shape)
        print("gesture:", ds["gesture"].shape)
        print("force unique:", set(ds["force"]))


if __name__ == "__main__":
    main()
