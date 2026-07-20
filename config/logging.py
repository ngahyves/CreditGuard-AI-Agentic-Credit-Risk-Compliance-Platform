# config/logging.py

import sys
import os
import json
import yaml
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 1. Automatic Path Detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

class JsonFormatter(logging.Formatter):
    """Custom logging class to return logs in JSON format."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        # Include exception traceback if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Include extra fields (like SK_ID_CURR)
        for key, value in record.__dict__.items():
            if key not in log_record and not key.startswith("_"):
                log_record[key] = value
        return json.dumps(log_record)

def get_logger(name: str):
    """
    Initialize a JSON logger.
    """
    # 2. Load YAML configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # 3. Setup log directory relative to PROJECT_ROOT
    logs_dir = PROJECT_ROOT / config["data_paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / "pipeline.log"
    log_level = config["logging"]["level"].upper()

    # 4. Setup Logger instance
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers when re-running notebook cells
    if logger.hasHandlers():
        return logger

    formatter = JsonFormatter()

    # 5. Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 6. File Handler (Rotating)
    fh = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger