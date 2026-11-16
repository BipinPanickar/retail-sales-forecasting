"""Simple logger writing to console and logs/pipeline.log."""
import logging
from src.utils.config import PROJECT_ROOT


def get_logger(name: str = "retail") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    for h in (logging.StreamHandler(), logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
