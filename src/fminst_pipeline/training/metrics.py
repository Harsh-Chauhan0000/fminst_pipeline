import torch
from fminst_pipeline.exception import CustomException
import sys

class AccuracyMetrics:
    def __init__(self):
        self.correct = 0.0
        self.total = 0.0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        try:
            _, predicted = torch.max(outputs.data, 1)
            self.correct += (predicted == labels).sum().item()
            self.total += labels.size(0)
        except Exception as e:
            raise CustomException(f"Accuracy update failed - {str(e)}", sys)

    def compute(self) -> float:
        try:
            if self.total == 0:
                return 0.0
            return self.correct / self.total
        except Exception as e:
            raise CustomException(f"Accuracy calculation failed - {str(e)}", sys)