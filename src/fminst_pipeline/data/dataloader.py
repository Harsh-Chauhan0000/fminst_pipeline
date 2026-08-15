import sys
import logging
from torch.utils.data import DataLoader, Dataset
from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException

logger = logging.getLogger(__name__)

def create_train_dataloaders(trainset: Dataset, valset: Dataset, config: Config)-> \
        tuple[DataLoader, DataLoader]:
    try:
        logger.info("Creating dataloaders")
        
        train_loader = DataLoader(
            trainset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=True,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logger.info("Successfully created train dataloader")

        val_loader = DataLoader(
            valset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=False,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logger.info("Successfully created val dataloader")

        return train_loader, val_loader
    except Exception as e:
        raise CustomException(e, sys)


def create_test_dataloader(testset: Dataset, config: Config)-> DataLoader:
    try:
        logger.info("Creating test dataloader")

        test_loader = DataLoader(
            testset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=False,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logger.info("Successfully created test dataloader")

        return test_loader
    except Exception as e:
        raise CustomException(e, sys)