import torch.nn as nn
from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException
import sys

def create_loss(config: Config) -> nn.Module:
    try:
        loss_fn = None

        if config.train.loss == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif config.train.loss == "bce_with_logits":
            loss_fn = nn.BCEWithLogitsLoss()
        elif config.train.loss == "mse":
            loss_fn = nn.MSELoss()
        elif config.train.loss == "binary_cross_entropy":
            loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss function: {config.train.loss}")
        
        return loss_fn

    except Exception as e:
        raise CustomException(f"Loss function creation failed - {str(e)}", sys)