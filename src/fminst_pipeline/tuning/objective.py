from fminst_pipeline.entity import artifact
import logging

import optuna
import torch
import copy

from fminst_pipeline.orchestrators.train import training_pipeline
from fminst_pipeline.entity.config import Config, load_config
from fminst_pipeline.entity.artifact import Artifact
from fminst_pipeline.utils.paths import ARTIFACT_DIR
from fminst_pipeline.tuning.search_space import hyperparam_space

class Objective:
    def __init__(self, base_config: Config, artifact: Artifact):
        self.base_config = base_config
        self.artifact = artifact
    
    def __call__(self, trial: optuna.Trial) -> float:
        config = copy.deepcopy(self.base_config)
        params = hyperparam_space(trial)
        
        config.train.optimizer.name = params["optimizer"]
        config.train.optimizer.learning_rate = params["learning_rate"]
        config.train.optimizer.weight_decay = params["weight_decay"]
        config.model.dropout = params["dropout"]
        config.data.dataloader.batch_size = params["batch_size"]
        
        history = training_pipeline(config, self.artifact)
        best_val_accuracy = max(history["val_accuracy"])
        trial.report(best_val_accuracy, step=len(history["val_accuracy"]))
        
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
        
        return best_val_accuracy
    


