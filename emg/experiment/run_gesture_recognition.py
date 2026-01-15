"""
EMG Gesture Recognition Experiment Runner

This script ONLY orchestrates the existing EMG pipeline.
NO signal processing / feature extraction / model definition here.
"""

from emg.dataset.build_dataset import build_dataset

from emg.preprocessing.emg_preprocessing_pipeline import (
    run_emg_preprocessing
)

from emg.feature.feature_extraction_pipeline import (
    run_feature_extraction
)

from emg.model.dataset.gesture_dataset_builder import (
    build_single_force_dataset,
    build_stratified_cross_force_dataset,
    build_device_specific_dataset
)

from emg.model.classical.svm import SVMClassifier
from emg.model.deep_learning.cnn import EMGCNN
from emg.model.deep_learning.lstm import EMGLSTM

from emg.experiment.utils import (
    split_train_test,
    evaluate_model
)

from emg.experiment.report.reporter import ExperimentReporter
from emg.experiment.report.visualizer import ExperimentVisualizer


def run_experiment(config):
    """
    Execute the complete EMG gesture recognition pipeline
    using EXISTING project modules.
    """

    # =====================================================
    # 1. Raw EMG → Dataset (h5 reading inside builder)
    # =====================================================
    dataset = build_emg_dataset(
        data_root=config["data_root"],
        subjects=config["subjects"],
        gestures=config["gestures"]
    )

    # =====================================================
    # 2. EMG preprocessing (band-pass + notch)
    # =====================================================
    dataset = run_emg_preprocessing(
        dataset,
        fs=config["fs"],
        bandpass=config["bandpass"],
        notch=config["notch"]
    )

    # =====================================================
    # 3. Feature extraction (time + frequency domain)
    # =====================================================
    dataset = run_feature_extraction(
        dataset,
        feature_list=config["features"],
        window_size=config["window_size"],
        window_step=config["window_step"]
    )

    results = {}

    # =====================================================
    # 4. Experiment loops
    # =====================================================
    for device_mode in config["device_modes"]:

        results[device_mode] = {}

        # ---------- Setting A: single-force ----------
        for f in config["force_levels"]:

            train_ds = build_single_force_dataset(dataset, f)
            train_ds = build_device_specific_dataset(train_ds, device_mode)

            for test_f in config["force_levels"]:
                test_ds = build_single_force_dataset(dataset, test_f)
                test_ds = build_device_specific_dataset(test_ds, device_mode)

                results[device_mode][(f, test_f)] = run_models(
                    train_ds, test_ds, config
                )

        # ---------- Setting B: cross-force ----------
        train_ds = build_stratified_cross_force_dataset(
            dataset,
            config["force_levels"],
            config["samples_per_force"]
        )
        train_ds = build_device_specific_dataset(train_ds, device_mode)

        for test_f in config["force_levels"]:
            test_ds = build_single_force_dataset(dataset, test_f)
            test_ds = build_device_specific_dataset(test_ds, device_mode)

            results[device_mode][("B", test_f)] = run_models(
                train_ds, test_ds, config
            )

    return results


def run_models(train_ds, test_ds, config):
    """
    Execute multiple gesture recognition models.
    """

    X_train, y_train, X_test, y_test = split_train_test(
        train_ds, test_ds
    )

    model_results = {}

    # ---- SVM ----
    svm = SVMClassifier(**config["svm"])
    model_results["SVM"] = evaluate_model(
        svm, X_train, y_train, X_test, y_test
    )

    # ---- CNN ----
    cnn = EMGCNN(
        n_channels=X_train.shape[1],
        n_classes=len(set(y_train)),
        **config["cnn"]
    )
    model_results["CNN"] = evaluate_model(
        cnn, X_train, y_train, X_test, y_test
    )

    # ---- LSTM ----
    lstm = EMGLSTM(
        input_size=X_train.shape[2],
        num_classes=len(set(y_train)),
        **config["lstm"]
    )
    model_results["LSTM"] = evaluate_model(
        lstm, X_train, y_train, X_test, y_test
    )

    return model_results


if __name__ == "__main__":

    experiment_config = {
        "data_root": "./data",
        "subjects": ["S1", "S2"],
        "gestures": list(range(1, 9)),

        "fs": 1000,
        "bandpass": (20, 450),
        "notch": 50,

        "window_size": 200,
        "window_step": 100,
        "features": [
            "MAV", "RMS", "WL", "ZC",
            "AR", "FFT_MeanFreq"
        ],

        "force_levels": [10, 20, 30, 40, 50],
        "samples_per_force": 100,
        "device_modes": ["dev1", "dev2", "fusion"],

        "svm": {"kernel": "rbf", "C": 1.0},
        "cnn": {"hidden_channels": 64},
        "lstm": {"hidden_size": 64}
    }

    # Run experiment
    results = run_experiment(experiment_config)

    # Report generation
    report_dir = "./outputs/experiment_01"

    reporter = ExperimentReporter(report_dir)
    df = reporter.save(results)

    # Visualization generation
    visualizer = ExperimentVisualizer(report_dir)
    visualizer.generate_all(df)
