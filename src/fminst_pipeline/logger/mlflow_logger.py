import logging
from typing import Any

import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)

class MLFlowLogger:
    def __init__(self, config):
        self.experiment_name = config.mlflow.experiment_name
        self.tracking_uri = config.mlflow.tracking_uri
        mlflow.set_experiment(self.experiment_name)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.run = None
    
    def start_run(self, run_name: str) -> None:
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLFlow run started with name: {run_name}")
    
    def end_run(self) -> None:
        if self.run:
            mlflow.end_run()
            self.run = None
        logger.info(f"MLFlow run ended")
    
    def log_params(self, params: dict[str, Any]) -> None:
        if self.run:
            mlflow.log_params(params)
        logger.info(f"MLFlow logged parameters: {params}")
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self.run:
            mlflow.log_metrics(metrics, step=step)
        logger.info(f"MLFlow logged metrics: {metrics} at step {step}")
