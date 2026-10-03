"""Logging setup: technical details go to the server console, never to the user."""
import logging

from app.config import settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL, format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
