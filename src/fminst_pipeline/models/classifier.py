import logging
import sys
from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException

import torch.nn as nn

logger = logging.getLogger(__name__)

def create_classifier(config: Config, backbone_out_features: int) -> nn.Module:

    try:
        model = nn.Sequential(
            nn.Linear(backbone_out_features, backbone_out_features//2),
            nn.BatchNorm1d(backbone_out_features//2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(backbone_out_features//2, backbone_out_features//4),
            nn.BatchNorm1d(backbone_out_features//4),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(backbone_out_features//4, config.model.num_classes)
        )
        for param in model.parameters():
            param.requires_grad =True
            
        logger.info(f"Created classifier for backbone: {config.model.model_backbone}")
        return model
    except Exception as e:
        raise CustomException(f"Failed to create classifier - {str(e)}", sys)