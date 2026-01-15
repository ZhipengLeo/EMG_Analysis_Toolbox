import os
import json
import pandas as pd


class ExperimentReporter:
    """
    Collect, serialize and export experiment results.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def results_to_dataframe(self, results):
        """
        Convert nested result dict to flat DataFrame.

        results[device][(train_force, test_force)][model] = acc
        """

        rows = []

        for device, device_res in results.items():
            for setting, model_res in device_res.items():

                train_force, test_force = setting

                for model_name, acc in model_res.items():
                    rows.append({
                        "device": device,
                        "train_force": train_force,
                        "test_force": test_force,
                        "model": model_name,
                        "accuracy": acc
                    })

        return pd.DataFrame(rows)

    def save(self, results):
        """
        Save results as CSV and JSON.
        """

        df = self.results_to_dataframe(results)

        csv_path = os.path.join(self.output_dir, "results.csv")
        json_path = os.path.join(self.output_dir, "results.json")

        df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[REPORT] CSV saved to {csv_path}")
        print(f"[REPORT] JSON saved to {json_path}")

        return df
