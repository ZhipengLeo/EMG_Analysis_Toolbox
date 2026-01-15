import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ExperimentVisualizer:
    """
    Visualization utilities for EMG experiment results.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_force_generalization_heatmap(self, df, device, model):
        """
        Heatmap: train force × test force
        """

        data = df[
            (df["device"] == device) &
            (df["model"] == model)
        ]

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
        plt.title(f"{model} – {device} Force Generalization")
        plt.xlabel("Test Force (%)")
        plt.ylabel("Train Force (%)")

        path = os.path.join(
            self.output_dir,
            f"heatmap_{device}_{model}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_model_comparison(self, df, device):
        """
        Bar chart: model comparison averaged over forces
        """

        data = df[df["device"] == device]

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
        plt.title(f"Model Comparison – {device}")
        plt.ylabel("Average Accuracy")

        path = os.path.join(
            self.output_dir,
            f"model_comparison_{device}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def generate_all(self, df):
        """
        Generate all standard plots.
        """

        for device in df["device"].unique():
            self.plot_model_comparison(df, device)

            for model in df["model"].unique():
                self.plot_force_generalization_heatmap(
                    df, device, model
                )
