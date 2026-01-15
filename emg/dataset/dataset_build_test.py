import numpy as np
from emg.dataset.build_dataset import build_dataset


def main():
    print(">>> Building EMG dataset...")

    dataset = build_dataset(
        root="F:\跨力手势分类实验20260113\EMG_Data",
        window_ms=100,
        step_ms=50,
    )

    X = dataset["X"]
    subject = dataset["subject"]
    collector = dataset["collector"]
    file_id = dataset["file"]

    print("\n=== Dataset summary ===")
    print(f"X shape        : {X.shape}  (N, C, F)")
    print(f"subject shape  : {subject.shape}")
    print(f"collector shape: {collector.shape}")
    print(f"file shape     : {file_id.shape}")

    print("\n=== Basic sanity checks ===")
    print(f"X dtype        : {X.dtype}")
    print(f"X min / max    : {X.min():.4f} / {X.max():.4f}")
    print(f"X mean / std   : {X.mean():.4f} / {X.std():.4f}")

    print("\n=== Subject distribution ===")
    unique_subjects, counts = np.unique(subject, return_counts=True)
    for s, c in zip(unique_subjects, counts):
        print(f"  subject {s}: {c} windows")

    print("\n>>> Dataset build check PASSED ✅")


if __name__ == "__main__":
    main()
