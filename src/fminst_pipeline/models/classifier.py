import logging

from fminst_pipeline.entity.config import Config

import torch.nn as nn

logger = logging.getLogger(__name__)

def create_classifier(config: Config, backbone_out_features: int) -> nn.Module:

    if config.model.model_backbone == "resnet50":
        model = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, config.model.num_classes)
        )
        for param in model.parameters():
            param.requires_grad =True
            
        logger.info(f"Created classifier for backbone: {config.model.model_backbone}")
        return model
    else:
        logger.error(f"Unknown backbone: {config.model.model_backbone}")
        raise ValueError(f"Unknown backbone: {config.model.model_backbone}")