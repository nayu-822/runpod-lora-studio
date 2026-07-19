from __future__ import annotations

import logging

LOGGER_NAME = "runpod_lora_studio"
_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        # Do not log API keys, tokens, cookies, headers, or rclone credentials.
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(logger.level)
    return logger
