import logging
from typing import Any

import torch
import torch.nn as nn
from fminst_pipeline.training.losses import create_loss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
    f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from fminst_pipeline.entity.config import Config

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: torch.device, config:Config, class_names: list[str],):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.class_names = class_names
        self.loss_fn = create_loss(self.config)

        self.model.to(self.device)

    @torch.no_grad()
    def predict(self) -> tuple[list[int], list[int]]:
        try:
            self.model.eval()
            all_labels = []
            all_predictions = []

            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images = images.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
            
            return all_labels, all_predictions
        except Exception as e:
            logger.error(f"Prediction failed - {str(e)}")
            raise e
    
    def evaluate(self) -> dict[str, Any]:
        try:
            all_labels, all_predictions = self.predict()

            loss = self.loss_fn(all_predictions, all_labels).item()
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            cm = confusion_matrix(all_labels, all_predictions)
            report = classification_report(all_labels, all_predictions, target_names=self.class_names, zero_division=0)
            
            results = {
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
                "report": report
            }
            logger.info(f"Evaluation completed! Loss: {loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
            logger.info(f"Classification Report - \n{report}")
            
            return results
        except Exception as e:
            logger.error(f"Evaluation failed - {str(e)}")
            raise e