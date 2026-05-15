import logging
import sys
from pathlib import Path
from typing import Optional

import structlog

from .config import LOG_LEVEL, LOG_FILE

_logging_configured = False

# Processors
_shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def setup_logging(
    level: str = LOG_LEVEL, log_file: Optional[Path] = LOG_FILE, console: bool = True
) -> None:
    global _logging_configured

    if _logging_configured and level == LOG_LEVEL:
        return

    log_level = getattr(logging, level.upper())

    # Configure structlog
    structlog.configure(
        processors=_shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # JSON formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=_shared_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    if console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _logging_configured = True


def get_logger(name: str):
    if not _logging_configured:
        setup_logging()

    if name.startswith("src."):
        name = name.replace("src.", "calmsense.")
    elif not name.startswith("calmsense"):
        name = f"calmsense.{name}"

    return structlog.get_logger(name)


class LoggerMixin:
    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_function_call(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug("entering", function=func.__name__)
        try:
            result = func(*args, **kwargs)
            logger.debug("exiting", function=func.__name__)
            return result
        except Exception as e:
            logger.error("error", function=func.__name__, error=str(e), exc_info=True)
            raise

    return wrapper


def log_exception(logger):
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception("exception", function=func.__name__, error=str(e))
                raise

        return wrapper

    return decorator
