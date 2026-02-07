import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, LOG_FILE

_logging_configured = False


def setup_logging(
    level: str = LOG_LEVEL, log_file: Optional[Path] = LOG_FILE, console: bool = True
) -> None:
    global _logging_configured

    if _logging_configured and level == LOG_LEVEL:
        return

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("calmsense")
    root_logger.setLevel(getattr(logging, level.upper()))

    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logging_configured = True
    root_logger.debug("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    if not _logging_configured:
        setup_logging()

    # Normalize namespace
    if name.startswith("src."):
        name = name.replace("src.", "calmsense.")
    elif not name.startswith("calmsense"):
        name = f"calmsense.{name}"

    return logging.getLogger(name)


class LoggerMixin:
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_function_call(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


def log_exception(logger: logging.Logger):
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator
