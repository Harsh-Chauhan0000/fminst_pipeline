import logging
import torch
import sys
from torchvision.datasets import FashionMNIST
from fminst_pipeline.entity.config import Config, load_config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.data.dataset import create_test_dataset
from fminst_pipeline.data.dataloader import create_test_dataloader
from fminst_pipeline.evaluation.evaluator import Evaluator
from fminst_pipeline.evaluation.plots import plot_confusion_matrix
from fminst_pipeline.logger import configure_logger
from fminst_pipeline.exception import CustomException
from fminst_pipeline.utils.utils import get_device, setup_seed
from fminst_pipeline.utils.paths import ARTIFACT_DIR

def evaluation_pipeline(config: Config, artifact: Artifact) -> None:
    try:
        logger = configure_logger()
        logger.info("=" * 60)
        logger.info("Starting Fashion-MNIST evaluation pipeline")
        logger.info("=" * 60)

        setup_seed(config)
        device = get_device(config)
        logger.info(f"Using device: {device}")

        logger.info("Loading test dataset")
        testset: FashionMNIST = create_test_dataset(config)
        test_loader = create_test_dataloader(testset, config)

        class_names = list(testset.classes)

        logger.info("Loading model with best weights")
        model = artifact.load_checkpoint("best_model.pt")

        logger.info("Initializing evaluator")
        evaluator = Evaluator(model=model, dataloader=test_loader, class_names=class_names, 
                            config=config, device=device)

        logger.info("Running evaluation")
        metrics = evaluator.evaluate()
        logger.info("Evaluation completed")
        artifact.save_json_report(data=metrics, filename="eval_metrics.json")
        logger.info("Saved evaluation metrics")
        
    except Exception as e:
        raise CustomException(e, sys)