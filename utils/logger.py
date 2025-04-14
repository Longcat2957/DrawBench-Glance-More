import logging
import colorlog


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)-8s] (%(name)s) - %(message)s <%(asctime)s>",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
