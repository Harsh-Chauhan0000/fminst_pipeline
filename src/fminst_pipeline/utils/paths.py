from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)