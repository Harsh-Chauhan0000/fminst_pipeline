from fminst_pipeline.entity.config import Config
import logging

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

def create_backbone(config: Config) -> tuple[nn.Module, int]:

    if config.model.model_backbone == "resnet50":
        if config.model.pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        
        logger.info(f"Created {config.model.model_backbone} backbone with pretrained: {config.model.pretrained}")
    else:
        logger.error(f"Unknown backbone: {config.model.model_backbone}")
        raise ValueError(f"Unknown backbone: {config.model.model_backbone}")

    return model, in_features