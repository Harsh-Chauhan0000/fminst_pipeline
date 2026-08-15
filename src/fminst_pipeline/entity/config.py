import os
from pydantic import BaseModel
from typing import Literal

from fminst_pipeline.utils.utils import load_yaml
from fminst_pipeline.utils.paths import CONFIG_DIR

# Base COnfig

class ProjectDescriptionConfig(BaseModel):
    name: str
    experiment_name: str

class LogConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class MLFlowConfig(BaseModel):
    enabled: bool
    experiment_name: str
    tracking_uri: str

class BaseConfig(BaseModel):
    project_description: ProjectDescriptionConfig
    seed: int
    device: Literal["auto", "cpu", "cuda"]
    logging: LogConfig
    mlflow: MLFlowConfig

# Data Config

class DatasetConfig(BaseModel):
    name: str
    root: str
    download: bool
    num_classes: int

class DataSplitConfig(BaseModel):
    validation_size: float
    random_state: int


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool

class DataResnetConfig(BaseModel):
    image_size: int
    centre_crop_size: int
    image_channel: int
    normalization_mean: list[float]
    normalization_std: list[float]

class Augumentation(BaseModel):
    enabled: bool
    horizontal_flip: dict
    rotation: dict

class DataConfig(BaseModel):
    dataset: DatasetConfig
    split: DataSplitConfig
    dataloader: DataLoaderConfig
    augmentation: Augumentation
    resnet: DataResnetConfig
    

# Model Config

class ModelConfig(BaseModel):
    model_backbone: str
    pretrained: bool
    num_classes: int
    dropout: float
    freeze_backbone: bool

# TRaining config

class OptimizerConfig(BaseModel):
    name: str
    learning_rate: float
    weight_decay: float

class SchedulerConfig(BaseModel):
    enabled: bool
    name: str
    min_lr: float

class EarlyStoppingConfig(BaseModel):
    enabled: bool
    patience: int
    monitor: str
    mode: str
    min_delta: float

class CheckpointConfig(BaseModel):
    save_best: bool
    save_last: bool
    monitor: str
    mode: str

class TrainingConfig(BaseModel):
    epochs: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: str
    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig
    baseline_freeze: bool

class Config(BaseModel):
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    train: TrainingConfig

def load_config() -> Config:

    raw_config = {
        "base": load_yaml(os.path.join(CONFIG_DIR,"base.yaml")),
        "data": load_yaml(os.path.join(CONFIG_DIR,"data.yaml")),
        "model": load_yaml(os.path.join(CONFIG_DIR,"model.yaml")),
        "train": load_yaml(os.path.join(CONFIG_DIR,"train.yaml")),
    }

    return Config(
        base=raw_config["base"],
        data=raw_config["data"],
        model=raw_config["model"],
        train=raw_config["train"]
    )