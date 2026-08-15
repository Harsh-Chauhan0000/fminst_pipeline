import logging
import torch
from fminst_pipeline.entity.config import Config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.exception import CustomException
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

class Callback:
    def __init__(self, config:Config, artifact:Artifact):
        self.config = config
        self.artifact = artifact
        self.patience_counter = 0
        self.best_metric_checkpoint = float('-inf') if self.config.train.checkpoint.mode.lower() == "max" else float('inf')
        self.best_metric_early_stopping = float('-inf') if self.config.train.early_stopping.mode.lower() == "max" else float('inf')
    
    def _save_checkpoint(self, epoch, model, optimizer, metrics: dict, save_best: bool = True, \
                                    save_last: bool = True)->None:
        mode = self.config.train.checkpoint.mode.lower()
        monitor = self.config.train.checkpoint.monitor.lower()
        try:
            checkpoint = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": metrics,
                        }

            if save_best:
                if mode == "max":
                    if metrics[monitor] > self.best_metric_checkpoint:
                        self.best_metric_checkpoint = metrics[monitor]
                        filename = "best_model.pth"
                        self.artifact.save_checkpoint(checkpoint=checkpoint, filename = filename)
                        logger.info(f"Saved best model to {filename}")
                elif mode == "min":
                    if metrics[monitor] < self.best_metric_checkpoint:
                        self.best_metric_checkpoint = metrics[monitor]
                        filename = "best_model.pth"
                        self.artifact.save_checkpoint(checkpoint=checkpoint, filename = filename)
                        logger.info(f"Saved best model to {filename}")

            if save_last:
                filename = f"last_model_epoch_{epoch}.pth"
                self.artifact.save_checkpoint(checkpoint=checkpoint, filename = filename)
                logger.info(f"Saved last model to {filename}")
                
        except Exception as e:
            raise CustomException(f"Failed to save checkpoint - {str(e)}", sys)

    def _check_early_stopping(self, metrics:dict) -> bool:
        mode = self.config.train.early_stopping.mode.lower()
        monitor = self.config.train.early_stopping.monitor.lower()
        patience = self.config.train.early_stopping.patience
        min_delta = self.config.train.early_stopping.min_delta
        if mode == "max":
            if metrics[monitor] > self.best_metric_early_stopping + min_delta:
                self.best_metric_early_stopping = metrics[monitor]
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
                return self.patience_counter >= patience
        elif mode == "min":
            if metrics[monitor] < self.best_metric_early_stopping - min_delta:
                self.best_metric_early_stopping = metrics[monitor]
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
                return self.patience_counter >= patience
        else:
            raise ValueError(f"Unsupported early stopping mode: {mode}")

    def on_epoch_end(self, epoch: int, model, optimizer, metrics: dict):
        try:
            if self.config.train.checkpoint.save_best or self.config.train.checkpoint.save_last:
                self._save_checkpoint(epoch, model, optimizer, metrics, save_best=self.config.train.checkpoint.save_best, save_last=self.config.train.checkpoint.save_last)
            if self.config.train.early_stopping.enabled:
                return self._check_early_stopping(metrics)
            return False  
        except Exception as e:
            raise CustomException(f"Callback on_epoch_end - epoch: {epoch} - failed - {str(e)}", sys)
