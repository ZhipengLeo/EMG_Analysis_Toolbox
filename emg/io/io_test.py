from emg.io.loader import load_emg_dataset

data = load_emg_dataset(
    r"F:\跨力手势分类实验20260113\EMG_Data\S01"
)

for collector, samples in data.items():
    print(f"{collector}: {len(samples)} files")
    print(samples[0].emg.shape)
