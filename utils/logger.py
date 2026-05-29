import logging
from datetime import datetime

from config.settings import PROJECT_ROOT


def setup_exception_logger() -> logging.Logger:
    """
    Set up a file logger for warnings and errors.

    The log file is created lazily on the first message (delay=True), so no
    file is written on runs that log nothing.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("exception_logger")
    logger.setLevel(logging.WARNING)

    if logger.handlers:
        return logger

    log_dir = PROJECT_ROOT / "utils" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"exceptions_{datetime.now():%Y%m%d}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger

# Global logger instance
exception_logger = setup_exception_logger()