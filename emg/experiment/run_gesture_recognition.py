import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from emg.features.normalize import SubjectNormalizer
from emg.dataset.build_dataset import build_dataset
from emg.dataset.split_dataset import split_by_force_level

from emg.model.classical.svm import SVMClassifier
from emg.model.deep_learning.cnn import EMGCNN
from emg.model.deep_learning.lstm import EMGLSTM

from emg.experiment.report.reporter import ExperimentReporter
from emg.experiment.report.visualizer import ExperimentVisualizer


# ===========================
# 🔧 Experiment Configuration
# ===========================

DATA_ROOT = r"F:\跨力手势分类实验20260113\EMG_Data"
OUTPUT_DIR = "outputs/gesture_recognition"

WINDOW_MS = 100
STEP_MS = 50

FORCE_LEVELS = [10, 20, 30, 40]

DEVICE_GROUPS = {
    "dev0": [0],
    "dev1": [1],
    "dev0+dev1": [0, 1],
}

FEATURES_TO_RUN = [
    "ALL", "RMS", "MAV", "VAR", "AAC", "SSC", "TTP", "MF", "MPF"
]

MODELS = ["SVM", "CNN", "LSTM"]

EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# 🔑 Dataset Mask
# ===========================

def apply_mask(dataset: dict, mask: np.ndarray) -> dict:
    masked = {}
    for k, v in dataset.items():
        if k == "single_feature_data":
            masked[k] = {feat: data[mask] for feat, data in v.items()}
        else:
            masked[k] = v[mask]
    return masked


# ===========================
# 🔑 Label Remapping
# ===========================

def remap_labels(y_train, y_test, X_test):
    classes = np.unique(y_train)
    class_map = {c: i for i, c in enumerate(classes)}

    y_train_m = np.array([class_map[y] for y in y_train])

    valid_mask = np.isin(y_test, classes)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]
    y_test_m = np.array([class_map[y] for y in y_test])

    return y_train_m, y_test_m, X_test, valid_mask


# ===========================
# 🔥 Torch Training
# ===========================

def train_torch(model, X_train, y_train, X_test, y_test):
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)

    return (preds == y_test).float().mean().item()


# ===========================
# 🧠 LSTM Sequence Builder
# ===========================

def build_lstm_sequences(X, y, trial_ids):
    """
    return:
        X_pad: (N_trials, T_max, D)
        y_seq: (N_trials,)
        lengths: (N_trials,)
    """
    X_seq = []
    y_seq = []
    lengths = []

    for tid in np.unique(trial_ids):
        mask = trial_ids == tid

        X_trial = X[mask]        # (T_i, C, F)
        y_trial = y[mask]

        assert np.all(y_trial == y_trial[0])

        T, C, F = X_trial.shape
        X_trial = X_trial.reshape(T, C * F)

        X_seq.append(X_trial)
        y_seq.append(y_trial[0])
        lengths.append(T)

    T_max = max(lengths)
    D = X_seq[0].shape[1]

    X_pad = np.zeros((len(X_seq), T_max, D), dtype=np.float32)

    for i, seq in enumerate(X_seq):
        X_pad[i, :seq.shape[0], :] = seq

    return X_pad, np.array(y_seq), np.array(lengths)

def train_torch_lstm(model, Xtr, ytr, len_tr, Xte, yte, len_te):
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xtr = torch.tensor(Xtr).to(DEVICE)
    ytr = torch.tensor(ytr).long().to(DEVICE)
    len_tr = torch.tensor(len_tr).to(DEVICE)

    Xte = torch.tensor(Xte).to(DEVICE)
    yte = torch.tensor(yte).long().to(DEVICE)
    len_te = torch.tensor(len_te).to(DEVICE)

    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xtr, len_tr), ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = torch.argmax(model(Xte, len_te), dim=1)

    return (pred == yte).float().mean().item()


# ===========================
# 🚀 Run Models
# ===========================

def run_models(
    X_train, y_train,
    X_test, y_test,
    trial_train, trial_test
):
    results = {}

    # ---------- SVM ----------
    if "SVM" in MODELS:
        svm = SVMClassifier()
        svm.fit(X_train, y_train)
        results["SVM"] = accuracy_score(y_test, svm.predict(X_test))

    # ---------- Label remap ----------
    y_train_m, y_test_m, X_test, valid_mask = remap_labels(
        y_train, y_test, X_test
    )
    trial_test = trial_test[valid_mask]

    if len(y_test_m) == 0:
        return results

    n_channels = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(np.unique(y_train_m))

    # ---------- CNN ----------
    if "CNN" in MODELS:
        cnn = EMGCNN(n_channels=n_channels, n_classes=n_classes)
        results["CNN"] = train_torch(
            cnn, X_train, y_train_m, X_test, y_test_m
        )

    # ---------- LSTM (trial-level) ----------
    if "LSTM" in MODELS:
        Xtr_seq, ytr_seq, len_tr = build_lstm_sequences(
            X_train, y_train_m, trial_train
        )
        Xte_seq, yte_seq, len_te = build_lstm_sequences(
            X_test, y_test_m, trial_test
        )

        _, T, D = Xtr_seq.shape  # D = C * F

        lstm = EMGLSTM(
            input_size=D,
            hidden_size=64,
            num_classes=n_classes
        )

        results["LSTM"] = train_torch_lstm(
            lstm,
            Xtr_seq,
            ytr_seq,
            len_tr,
            Xte_seq,
            yte_seq,
            len_te,
        )

    return results


# ===========================
# 🧠 Main Pipeline
# ===========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] Building dataset...")
    dataset = build_dataset(DATA_ROOT, WINDOW_MS, STEP_MS)

    dataset = apply_mask(
        dataset, np.isin(dataset["force"], FORCE_LEVELS)
    )

    results = {}

    for device_name, device_ids in DEVICE_GROUPS.items():
        print(f"\n[DEVICE] {device_name}")
        results[device_name] = {}

        dset = apply_mask(
            dataset, np.isin(dataset["collector"], device_ids)
        )

        # ======================================================
        # A️⃣ single-force → different-force
        # ======================================================
        for train_force in FORCE_LEVELS:
            for test_force in FORCE_LEVELS:
                if train_force == test_force:
                    continue

                split = split_by_force_level(
                    dset,
                    test_force=test_force,
                    train_forces_mixed=(train_force,),
                )

                ytr = split["train_cross"]["gesture"]
                yte = split["test"]["gesture"]

                if len(np.unique(ytr)) < 2:
                    continue

                key = (train_force, test_force)
                results[device_name][key] = {}

                for feat in FEATURES_TO_RUN:
                    Xtr = (
                        split["train_cross"]["X"]
                        if feat == "ALL"
                        else split["train_cross"]["single_feature_data"][feat]
                    )
                    Xte = (
                        split["test"]["X"]
                        if feat == "ALL"
                        else split["test"]["single_feature_data"][feat]
                    )

                    normalizer = SubjectNormalizer()

                    results[device_name][key][feat] = run_models(
                        normalizer.fit_transform(Xtr),
                        ytr,
                        normalizer.transform(Xte),
                        yte,
                        split["train_cross"]["trial"],
                        split["test"]["trial"],
                    )

        # ======================================================
        # B️⃣ mixed-force (balanced)
        # ======================================================
        for test_force in FORCE_LEVELS:
            split = split_by_force_level(
                dset,
                test_force=test_force,
                train_forces_mixed=tuple(
                    f for f in FORCE_LEVELS if f != test_force
                ),
                balanced_split=True,
            )

            ytr = split["train_mixed_balanced"]["gesture"]
            yte = split["test"]["gesture"]

            key = ("mixed_balanced", test_force)
            results[device_name][key] = {}

            for feat in FEATURES_TO_RUN:
                Xtr = (
                    split["train_mixed_balanced"]["X"]
                    if feat == "ALL"
                    else split["train_mixed_balanced"]["single_feature_data"][feat]
                )
                Xte = (
                    split["test"]["X"]
                    if feat == "ALL"
                    else split["test"]["single_feature_data"][feat]
                )

                normalizer = SubjectNormalizer()

                results[device_name][key][feat] = run_models(
                    normalizer.fit_transform(Xtr),
                    ytr,
                    normalizer.transform(Xte),
                    yte,
                    split["train_mixed_balanced"]["trial"],
                    split["test"]["trial"],
                )

        # ======================================================
        # C️⃣ same-force baseline
        # ======================================================
        for test_force in FORCE_LEVELS:
            split = split_by_force_level(
                dset,
                test_force=test_force,
                train_forces_mixed=(),
            )

            ytr = split["train_same"]["gesture"]
            yte = split["test"]["gesture"]

            key = ("same", test_force)
            results[device_name][key] = {}

            for feat in FEATURES_TO_RUN:
                Xtr = (
                    split["train_same"]["X"]
                    if feat == "ALL"
                    else split["train_same"]["single_feature_data"][feat]
                )
                Xte = (
                    split["test"]["X"]
                    if feat == "ALL"
                    else split["test"]["single_feature_data"][feat]
                )

                normalizer = SubjectNormalizer()

                results[device_name][key][feat] = run_models(
                    normalizer.fit_transform(Xtr),
                    ytr,
                    normalizer.transform(Xte),
                    yte,
                    split["train_same"]["trial"],
                    split["test"]["trial"],
                )

    reporter = ExperimentReporter(OUTPUT_DIR)
    df = reporter.save(results)

    ExperimentVisualizer(OUTPUT_DIR).generate_all(df)

    print("\n✅ ALL EMG EXPERIMENTS FINISHED")


if __name__ == "__main__":
    main()
