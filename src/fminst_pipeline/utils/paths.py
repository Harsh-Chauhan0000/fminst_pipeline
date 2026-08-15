from pathlib import Path
import os
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs",TIMESTAMP)
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts",TIMESTAMP)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)