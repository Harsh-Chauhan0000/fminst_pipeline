from typing import Any
import os
import json
import shutil
import torch

class Artifact:

    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

        self.checkpoints_dir = os.path.join(artifact_dir, "checkpoints")
        self.plots_dir = os.path.join(artifact_dir, "plots")
        self.reports_dir = os.path.join(artifact_dir, "reports")
        self.predictions_dir = os.path.join(artifact_dir, "predictions")

        self._create_directories()

    def _create_directories(self) -> None:

        directories = [
            self.artifact_dir,
            self.checkpoints_dir,
            self.plots_dir,
            self.reports_dir,
            self.predictions_dir,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_checkpoint_path(self, filename: str) -> str:
        return os.path.join(self.checkpoints_dir, filename)

    def save_checkpoint(self, checkpoint: Any, filename: str) -> str:
        path = self.get_checkpoint_path(filename)
        torch.save(checkpoint, path)
        return path

    def get_plot_path(self, filename: str) -> str:
        return os.path.join(self.plots_dir, filename)

    def get_report_path(self, filename: str) -> str:
        return os.path.join(self.reports_dir, filename)

    def save_json_report(self, data: dict, filename: str) -> str:
        path = self.get_report_path(filename)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                default=str,
            )

        return path

    def get_prediction_path(self, filename: str) -> str:
        return os.path.join(self.predictions_dir, filename)

    def save_predictions(self, data: Any, filename: str) -> str:
        path = self.get_prediction_path(filename)

        if hasattr(data, "to_csv"):
            data.to_csv(path, index=False)
        else:
            raise TypeError("Predictions must be a pd df or supported objects.")

        return path

    def delete_all(self) -> None:

        if os.path.exists(self.artifact_dir):
            shutil.rmtree(self.artifact_dir)

        self._create_directories()

    def list_artifacts(self) -> list[str]:

        return [
            path
            for path in os.listdir(self.artifact_dir)
            if os.path.isfile(os.path.join(self.artifact_dir, path))
        ]