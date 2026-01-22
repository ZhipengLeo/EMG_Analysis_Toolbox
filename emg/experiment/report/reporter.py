import os
import json
import pandas as pd


def _stringify_keys(obj):
    """
    Recursively convert non-JSON-serializable dict keys (e.g. tuple)
    into strings, so that json.dump will not fail.
    """
    if isinstance(obj, dict):
        return {
            str(k): _stringify_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    else:
        return obj


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

        Expected structure:
        results[device][(train_force, test_force)][feature][model] = acc
        """

        rows = []

        for device, device_res in results.items():
            for setting, feature_res in device_res.items():

                # setting can be:
                #   (train_force, test_force)
                train_force, test_force = setting

                for feature_name, model_res in feature_res.items():
                    for model_name, acc in model_res.items():
                        rows.append({
                            "device": device,
                            "train_force": train_force,
                            "test_force": test_force,
                            "feature": feature_name,
                            "model": model_name,
                            "accuracy": acc
                        })

        return pd.DataFrame(rows)

    def save(self, results):
        """
        Save results as CSV and JSON.
        """

        # 1️⃣ CSV（论文 / 统计分析主力）
        df = self.results_to_dataframe(results)
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        # 2️⃣ JSON（完整实验记录，便于复现）
        json_safe_results = _stringify_keys(results)
        json_path = os.path.join(self.output_dir, "results.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_safe_results, f, indent=2)

        print(f"[REPORT] CSV saved to {csv_path}")
        print(f"[REPORT] JSON saved to {json_path}")

        return df
