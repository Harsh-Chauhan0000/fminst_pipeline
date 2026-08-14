import sys
import logging
from torch.utils.data import DataLoader, Dataset
from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException

logger = logging.getLogger(__name__)

def create_dataloaders(trainset: Dataset, valset: Dataset, testset: Dataset, config: Config)-> \
        tuple[DataLoader, DataLoader, DataLoader]:
    try:
        logging.info("Creating dataloaders")
        
        train_loader = DataLoader(
            trainset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=True,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logging.info("Successfully created train dataloader")

        val_loader = DataLoader(
            valset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=False,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logging.info("Successfully created val dataloader")

        test_loader = DataLoader(
            testset,
            batch_size=config.data.dataloader.batch_size,
            shuffle=False,
            num_workers=config.data.dataloader.num_workers,
            pin_memory=config.data.dataloader.pin_memory
        )
        logging.info("Successfully created test dataloader")

        return train_loader, val_loader, test_loader
    except Exception as e:
        raise CustomException(e, sys)