import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

from fminst_pipeline.utils.paths import LOG_DIR

MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Number of backup log files to keep

def configure_logger() -> logging.Logger:
    logger = logging.getLogger()

    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    log_file = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_file_path = os.path.join(LOG_DIR, log_file)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Create a rotating file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_FILE_SIZE,
                                    backupCount=BACKUP_COUNT, encoding = "utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()