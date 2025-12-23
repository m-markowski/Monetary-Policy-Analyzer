import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))
import logging
from datetime import datetime
from config.settings import PROJECT_ROOT

class LazyFileHandler(logging.Handler):
    """
    File handler that creates the log file only when the first message is logged.
    """
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._handler = None

    def emit(self, record):
        if self._handler is None:
            self.log_path.parent.mkdir(exist_ok=True)
            self._handler = logging.FileHandler(self.log_path, encoding='utf-8')
            self._handler.setFormatter(self.formatter)
            self._handler.setLevel(self.level)
        self._handler.emit(record)

    def close(self):
        if self._handler:
            self._handler.close()
        super().close()

def setup_exception_logger() -> logging.Logger:
    """
    Set up a simple logger for exceptions only.
    Log file is created only when the first exception is logged.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('exception_logger')
    logger.setLevel(logging.ERROR)

    if logger.handlers:
        return logger

    log_dir = PROJECT_ROOT / "utils" / "logs"
    timestamp = datetime.now().strftime('%Y%m%d')
    log_path = log_dir / f'exceptions_{timestamp}.log'

    file_handler = LazyFileHandler(log_path)
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Global logger instance
exception_logger = setup_exception_logger()