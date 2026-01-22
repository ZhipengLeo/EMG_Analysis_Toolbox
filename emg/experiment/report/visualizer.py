import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ExperimentVisualizer:
    """
    Visualization utilities for EMG experiment results.
    Now supports feature dimension.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_force_generalization_heatmap(self, df, device, model, feature):
        """
        Heatmap: train force × test force
        for a specific device / model / feature
        """

        data = df[
            (df["device"] == device) &
            (df["model"] == model) &
            (df["feature"] == feature)
        ]

        if data.empty:
            return

        pivot = data.pivot(
            index="train_force",
            columns="test_force",
            values="accuracy"
        )

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis"
        )
        plt.title(f"{model} – {device} – {feature}")
        plt.xlabel("Test Force (%)")
        plt.ylabel("Train Force (%)")

        path = os.path.join(
            self.output_dir,
            f"heatmap_{device}_{model}_{feature}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_model_comparison(self, df, device, feature):
        """
        Bar chart: model comparison averaged over forces
        for a specific feature
        """

        data = df[
            (df["device"] == device) &
            (df["feature"] == feature)
        ]

        if data.empty:
            return

        avg = (
            data
            .groupby("model")["accuracy"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=avg,
            x="model",
            y="accuracy"
        )
        plt.ylim(0, 1)
        plt.title(f"Model Comparison – {device} – {feature}")
        plt.ylabel("Average Accuracy")

        path = os.path.join(
            self.output_dir,
            f"model_comparison_{device}_{feature}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_feature_comparison(self, df, device, model):
        """
        Bar chart: feature comparison for a given model
        """

        data = df[
            (df["device"] == device) &
            (df["model"] == model)
        ]

        if data.empty:
            return

        avg = (
            data
            .groupby("feature")["accuracy"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(7, 4))
        sns.barplot(
            data=avg,
            x="feature",
            y="accuracy"
        )
        plt.ylim(0, 1)
        plt.title(f"Feature Comparison – {device} – {model}")
        plt.ylabel("Average Accuracy")

        path = os.path.join(
            self.output_dir,
            f"feature_comparison_{device}_{model}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def generate_all(self, df):
        """
        Generate all standard plots.
        """

        devices = df["device"].unique()
        models = df["model"].unique()
        features = df["feature"].unique()

        # 1️⃣ Model comparison per feature
        for device in devices:
            for feature in features:
                self.plot_model_comparison(df, device, feature)

        # 2️⃣ Force generalization heatmaps
        for device in devices:
            for model in models:
                for feature in features:
                    self.plot_force_generalization_heatmap(
                        df, device, model, feature
                    )

        # 3️⃣ Feature comparison per model
        for device in devices:
            for model in models:
                self.plot_feature_comparison(df, device, model)
