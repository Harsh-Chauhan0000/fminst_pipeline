from fminst_pipeline.entity.artifact import Artifact
import logging
import optuna
import sys

from fminst_pipeline.tuning.objective import Objective
from fminst_pipeline.entity.config import Config
from fminst_pipeline.exception import CustomException

logger = logging.getLogger(__name__)

def create_study(config:Config) -> optuna.Study:
    study = optuna.create_study(
        study_name=config.tuning.study.study_name,
        direction=config.tuning.study.direction,
        sampler=optuna.samplers.TPESampler(
            seed=config.base.seed
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )
    return study

def run_study(config: Config, artifact: Artifact):
    logger.info("Starting to run study")
    study = create_study(config)
    objective = Objective(config, artifact)
    try:
        study.optimize(objective, n_trials=config.tuning.study.n_trials)
        logger.info("Finished running study")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
    except Exception as e:
        logger.error(f"Error running study: {e}")
        raise CustomException(e, sys)
    return study
    
