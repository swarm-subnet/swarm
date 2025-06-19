import os
import logging
from logging.handlers import RotatingFileHandler
import bittensor as bt

EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10


def setup_events_logger(full_path, events_retention_size):
    logging.addLevelName(EVENTS_LEVEL_NUM, "EVENT")

    logger = logging.getLogger("event")
    logger.setLevel(EVENTS_LEVEL_NUM)

    def event(self, message, *args, **kws):
        if self.isEnabledFor(EVENTS_LEVEL_NUM):
            self._log(EVENTS_LEVEL_NUM, message, args, **kws)

    logging.Logger.event = event

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(full_path, "events.log"),
        maxBytes=events_retention_size,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(EVENTS_LEVEL_NUM)
    logger.addHandler(file_handler)

    return logger


class ColoredLogger:
    """A simple logger that uses ANSI colors when calling bt.logging methods."""

    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"
    CYAN = "cyan"
    MAGENTA = "magenta"
    WHITE = "white"
    PURPLE = "purple"
    GRAY = "gray"
    RESET = "reset"

    _COLORS = {
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "green": "\033[92m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "white": "\033[97m",
        "gray": "\033[90m",
        "reset": "\033[0m",
        "purple": "\033[35m",
    }

    @staticmethod
    def _colored_msg(message: str, color: str) -> str:
        """Return the colored message based on the color provided."""
        if color not in ColoredLogger._COLORS:
            # Default to no color if unsupported color is provided
            return message
        return (
            f"{ColoredLogger._COLORS[color]}{message}{ColoredLogger._COLORS['reset']}"
        )

    @staticmethod
    def info(message: str, color: str = "blue") -> None:
        bt.logging.info(ColoredLogger._colored_msg(message, color))

    @staticmethod
    def warning(message: str, color: str = "yellow") -> None:
        bt.logging.warning(ColoredLogger._colored_msg(message, color))

    @staticmethod
    def error(message: str, color: str = "red") -> None:
        bt.logging.error(ColoredLogger._colored_msg(message, color))

    @staticmethod
    def success(message: str, color: str = "green") -> None:
        bt.logging.success(ColoredLogger._colored_msg(message, color))
