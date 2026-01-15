from emg.io.loader import load_emg_dataset
from emg.preprocess.filter import preprocess_emg
from emg.preprocess.visualize import plot_emg_before_after

data = load_emg_dataset(r"F:\跨力手势分类实验20260113\EMG_Data\S01")

cid = sorted(data.keys())[0]     # 自动取第一个采集器
sample = data[cid][0]

raw_emg = sample.emg
fs = sample.fs

filtered_emg = preprocess_emg(raw_emg, fs)

plot_emg_before_after(
    raw_emg,
    filtered_emg,
    fs,
    channels=(0, ),
    title=f"Collector {cid}: Raw vs Filtered EMG"
)