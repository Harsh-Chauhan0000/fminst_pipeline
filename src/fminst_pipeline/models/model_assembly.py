import logging
import sys
from fminst_pipeline.entity.config import Config
from fminst_pipeline.models.backbone import create_backbone
from fminst_pipeline.models.classifier import create_classifier
from fminst_pipeline.exception import CustomException
import torch.nn as nn

logger = logging.getLogger(__name__)

def create_model(config: Config) -> nn.Module:
    logger.info("Creating model")

    try:
        model, in_features = create_backbone(config)

        if config.model.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

            logger.info("Backbone freezed")
    
        model.fc = create_classifier(config, in_features)

        logger.info(f"Created model: {model}")
        return model

    except Exception as e:
        raise CustomException(f"Model creation failed - {str(e)}", sys)
