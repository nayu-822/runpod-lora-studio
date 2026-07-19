from __future__ import annotations

import logging

from runpod_lora_studio.logging.config import LOGGER_NAME, configure_logging


def test_configure_logging_does_not_duplicate_handlers() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    try:
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()

        configure_logging("INFO")
        handler_count = len(logger.handlers)
        configure_logging("DEBUG")

        assert len(logger.handlers) == handler_count == 1
        assert logger.level == logging.DEBUG
        assert logger.handlers[0].level == logging.DEBUG
    finally:
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
