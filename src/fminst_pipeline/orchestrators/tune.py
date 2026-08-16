import logging
import sys
import json
import os
from fminst_pipeline.exception import CustomException
from fminst_pipeline.logger import configure_logger
from fminst_pipeline.entity.config import Config, load_config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.utils.paths import ARTIFACT_DIR, TUNING_DIR
from fminst_pipeline.tuning.study import run_study

def tuning_pipeline(config: Config, artifact: Artifact):
    logger = configure_logger()
    logger.info("=" * 60)
    logger.info("Starting Fashion-MNIST hyperparameter tuning pipeline")
    logger.info("=" * 60)
    try:
        logger.info("Running study")
        study = run_study(config, artifact)
        with open(os.path.join(TUNING_DIR, "best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)
        logger.info("Finished running study")
        return study
    except Exception as e:
        logger.error(f"Error running study: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config = load_config()
        artifact = Artifact(artifact_dir=ARTIFACT_DIR)
        tuning_pipeline(config, artifact)
    except Exception as e:
        raise CustomException(e, sys)