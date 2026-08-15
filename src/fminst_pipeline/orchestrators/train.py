import logging
import torch
import sys

from fminst_pipeline.entity.config import Config, load_config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.data.dataset import create_train_datasets
from fminst_pipeline.data.dataloader import create_train_dataloaders
from fminst_pipeline.exception import CustomException
from fminst_pipeline.utils.utils import get_device, setup_seed
from fminst_pipeline.models.model_assembly import create_model
from fminst_pipeline.training.trainer import Trainer
from fminst_pipeline.utils.paths import ARTIFACT_DIR
from fminst_pipeline.logger import configure_logger

def training_pipeline(config: Config, artifact: Artifact) -> None:
    try:
        logger = configure_logger()
        logger.info("=" * 60)
        logger.info("Starting Fashion-MNIST training pipeline")
        logger.info("=" * 60)

        setup_seed(config)

        device = get_device(config)
        logger.info(f"Using device: {device}")
        logger.info(f"Loading train and val datasets")
        trainset, valset = create_train_datasets(config)
        train_loader, val_loader = create_train_dataloaders(trainset, valset, config)
        logger.info(f"Loading model")
        model = create_model(config)
        logger.info(f"Initializing trainer")
        trainer = Trainer(model=model, train_loader=train_loader, 
                            val_loader=val_loader, config=config, 
                            artifact=artifact, device=device)

        logger.info("Starting training")
        history = trainer.fit()
        logger.info("Training completed")
        artifact.save_json_report(data=history, filename="train_history.json")
        logger.info("Saved train history")
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config = load_config()
        artifact = Artifact(artifact_dir=ARTIFACT_DIR)
        training_pipeline(config, artifact)
    except Exception as e:
        raise CustomException(e, sys)