from pathlib import Path
import os
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
trial_name = "trial_test_1"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs",trial_name)
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts",trial_name)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
TUNING_DIR = os.path.join(PROJECT_ROOT, "tuning",trial_name)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(TUNING_DIR, exist_ok=True)