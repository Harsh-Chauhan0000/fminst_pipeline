import torch
import torch.nn as nn
import sys

from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException

def create_optimizer(model:nn.Module, config:Config):
    lr = config.train.optimizer.learning_rate
    wd = config.train.optimizer.weight_decay
    try:
        if config.train.optimizer.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "adamax":
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
        elif config.train.optimizer.name.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {config.train.optimizer.name}")
        return optimizer
    except Exception as e:
        raise CustomException(f"Optimizer creation failed - {str(e)}", sys)
    

