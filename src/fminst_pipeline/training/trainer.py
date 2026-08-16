import logging
from typing import Dict
import sys
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fminst_pipeline.entity.config import Config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.training.losses import create_loss
from fminst_pipeline.training.optimizers import create_optimizer
from fminst_pipeline.training.metrics import AccuracyMetrics
from fminst_pipeline.training.callbacks import Callback
from fminst_pipeline.exception import CustomException
import sys

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self,model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                    device:torch.device, config:Config, artifact:Artifact):
        self.config = config
        self.artifact = artifact
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = create_optimizer(self.model, self.config)
        self.loss_fn = create_loss(self.config)
        self.metrics = AccuracyMetrics()
        self.device = device
        self.callback = Callback(self.config, self.artifact)
        self.model.to(self.device)
        
        logger.info(f"Trainer initialized for model: {model.__class__.__name__} on device: {self.device}")


    def train_epoch(self, epoch:int)-> Dict[str, float]:
        try:
            total = 0
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item() * target.size(0)
                self.metrics.update(outputs, target)

                total += target.size(0)

                if batch_idx % self.config.train.epochs == 0:
                    logger.info(f"Train - Epoch: {epoch} | {self.config.train.epochs}, Batch: {batch_idx}, Loss: {running_loss/total:.4f}, Accuracy: {self.metrics.compute():.4f}")
            
            epoch_loss = running_loss / total
            epoch_accuracy = self.metrics.compute()
            
            return {
                "loss": epoch_loss,
                "accuracy": epoch_accuracy
            }
        
        except Exception as e:
            raise CustomException(f"Training epoch failed - {str(e)}", sys)
        

    @torch.no_grad()
    def validate_epoch(self, epoch: int)-> Dict[str, float]:
        try:
            self.model.eval()
            total = 0
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)
                
                running_loss += loss.item() * target.size(0)
                self.metrics.update(outputs, target)

                total += target.size(0)

                if batch_idx % self.config.train.epochs == 0:
                    logger.info(f"Val - Epoch:{epoch} | {self.config.train.epochs}, Batch: {batch_idx}, Loss: {running_loss/total:.4f}, Accuracy: {self.metrics.compute():.4f}")
            
            epoch_loss = running_loss / total
            epoch_accuracy = self.metrics.compute()
            logger.info(f"Val - Epoch {epoch} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            return {
                "loss": epoch_loss,
                "accuracy": epoch_accuracy
            }
        
        except Exception as e:
            raise CustomException(f"Validation epoch failed - {str(e)}", sys)
        

    def fit(self)-> dict[str, list[float]]:
        try:
            history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
            logger.info(f"Starting training for {self.config.train.epochs} epochs")

            for epoch in range(1, self.config.train.epochs + 1):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate_epoch(epoch)
                metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"]
                }
                for key, value in metrics.items():
                    history[key].append(value)
                
                logger.info(f"Epoch {epoch} | {self.config.train.epochs} completed. Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
                stop_training = self.callback.on_epoch_end(epoch, self.model, self.optimizer, metrics)
                if stop_training:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            return history
        except Exception as e:
            raise CustomException(f"Training failed - {str(e)}", sys)
        

        
        
        
        
    
    
    