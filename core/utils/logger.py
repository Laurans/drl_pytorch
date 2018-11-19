import logging, coloredlogs
from logging import Logger


def loggerConfig(
    log_file: str, verbose: int, namelogger: str = "drl_pytorch"
) -> Logger:
    logger = logging.getLogger(namelogger)
    fmt = "[%(levelname)-8s] (%(module)s - %(funcName)s) %(message)s"

    logger.propagate = 0
    if not logger.handlers:

        formatter = logging.Formatter(fmt)

        fileHandler = logging.FileHandler(log_file, "w")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)

        if verbose >= 2:
            logger.setLevel(logging.DEBUG)
            coloredlogs.install(logger=logger, fmt=fmt)
        elif verbose >= 1:
            logger.setLevel(logging.INFO)
            coloredlogs.install(logger=logger, fmt=fmt)
        else:
            streamhandler.setLevel(logging.CRITICAL)
            fileHandler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)

    return logger
