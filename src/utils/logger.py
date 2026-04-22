import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger with:
    - StreamHandler  → stdout (always)
    - FileHandler    → log_file (optional)
    Format: [LEVEL] YYYY-MM-DD HH:MM:SS | module_name | message
    """
    
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    
    if log_file:
        Path(log_file).parent.mkdir(parents= True, exist_ok= True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
    