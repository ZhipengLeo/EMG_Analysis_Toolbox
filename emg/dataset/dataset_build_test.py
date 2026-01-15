from emg.io.loader import load_emg_dataset
from emg.dataset.emg_dataset import EMGDatasetBuilder


def main():

    subjects = load_emg_dataset("data/emg_data.h5")

    # 此时假设：
    # - subjects 已完成 preprocess
    # - 已完成 feature extraction + normalization
    # - trial.features 已就绪

    builder = EMGDatasetBuilder()
    dataset = builder.build(subjects)

    print("Dataset constructed:")
    print("X:", dataset["X"].shape)
    print("gesture:", dataset["gesture"].shape)
    print("force:", dataset["force"].shape)
    print("trial:", dataset["trial"].shape)


if __name__ == "__main__":
    main()
