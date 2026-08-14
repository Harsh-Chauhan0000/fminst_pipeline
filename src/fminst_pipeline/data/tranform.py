import sys
from torchvision import transforms
from collections.abc import Callable
from typing import Any
from fminst_pipeline.entity.config import Config

import logging
from fminst_pipeline.exception import CustomException

logger = logging.getLogger(__name__)

def train_transforms(config:Config):
    try:
        logger.info("Creating train transforms")
        transforms_list: list[Callable[[Any], Any]] = [
            transforms.Resize(config.data.resnet.image_size),
            transforms.CenterCrop(config.data.resnet.centre_crop_size),
            transforms.Grayscale(num_output_channels=config.data.resnet.image_channel)
        ]

        if config.data.augmentation.enabled:
            transforms_list.append(
                transforms.RandomHorizontalFlip(config.data.augmentation.horizontal_flip["probability"])
            )
            transforms_list.append(
                transforms.RandomRotation(config.data.augmentation.rotation["degrees"])
            )

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=config.data.resnet.normalization_mean,
                                                    std=config.data.resnet.normalization_std))

        logger.info("Successfully created train transforms")
        return transforms.Compose(transforms_list)
    except Exception as e:
        raise CustomException(e, sys)


def val_transforms(config:Config):
    try:
        logger.info("Creating val transforms")
        transforms_list = [
            transforms.Resize(config.data.resnet.image_size),
            transforms.CenterCrop(config.data.resnet.centre_crop_size),
            transforms.Grayscale(num_output_channels=config.data.resnet.image_channel),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.resnet.normalization_mean, std=config.data.resnet.normalization_std)
        ]
        logger.info("Successfully created val transforms")
        return transforms.Compose(transforms_list)
    except Exception as e:
        raise CustomException(e, sys)