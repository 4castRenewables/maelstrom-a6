import logging

logger = logging.getLogger(__name__)


def import_deep500() -> object:
    try:
        import deep500.utils.timer_torch as timer
    except ModuleNotFoundError:
        import unittest.mock as mock

        logger.exception(
            "Module 'deep500' not found, mocking deep500.utils.timer_torch "
            "module",
        )

        timer = mock.MagicMock()

    return timer
