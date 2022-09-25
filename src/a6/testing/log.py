import logging

logger = logging.getLogger(__name__)


def log_text(text: str) -> None:
    logger.info(text)
