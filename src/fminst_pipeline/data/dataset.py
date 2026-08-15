import sys
import torch
import logging
from torch.utils.data import Subset, Dataset
from torchvision.datasets import FashionMNIST

from fminst_pipeline.exception import CustomException
from fminst_pipeline.data.tranform import train_transforms, val_transforms
from fminst_pipeline.entity.config import Config

logger = logging.getLogger(__name__)

def create_train_datasets(config:Config)-> tuple[Dataset, Dataset]:

    try:
        logger.info("Starting to create datasets")
        train_transform = train_transforms(config)
        val_transform = val_transforms(config)
        logger.info("Created train and val transforms")

        trainbase = FashionMNIST(
            root=config.data.dataset.root,
            train=True,
            download=config.data.dataset.download,
            transform=train_transform
        )

        valbase = FashionMNIST(
            root=config.data.dataset.root,
            train=True,
            download=False,
            transform=val_transform
        )

        logger.info("Created train and val datasets")
    except Exception as e:
        raise CustomException(e, sys)
    
    dataset_len = len(trainbase)
    val_size = int(dataset_len * config.data.split.validation_size)
    train_size = dataset_len - val_size

    logger.info(f"Dataset split sizes: Train - {train_size}, Val - {val_size}")

    generator = torch.Generator().manual_seed(config.data.split.random_state)

    indices = torch.randperm(dataset_len, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    trainset = Subset(trainbase, train_indices)
    valset = Subset(valbase, val_indices)

    logger.info("Created trainset and valset")

    return trainset, valset

def create_test_dataset(config: Config)-> FashionMNIST:
    try:
        logger.info("Starting to create test dataset")
        val_transform = val_transforms(config)
        testset = FashionMNIST(
            root=config.data.dataset.root,
            train=False,
            download=config.data.dataset.download,
            transform=val_transform
        )
        logger.info("Created test dataset")
    except Exception as e:
        raise CustomException(e, sys)

    return testset

    