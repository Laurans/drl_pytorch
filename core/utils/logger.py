import logging, coloredlogs
from logging import Logger


def loggerConfig(
    log_file: str, verbose: int, namelogger: str = "drl_pytorch"
) -> Logger:
    """Config a logger to stream and write in file according to the level.

    Args:
        log_file (str): Log file to write log in it
        verbose (int): verbosity -> 2 for debug (stream + file), 1 for info (stream + file), 0 to shut stream log
        namelogger (str, optional): Defaults to "drl_pytorch". Logger name to get unique logger

    Returns:
        Logger: Logger object
    """

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
            coloredlogs.install(logger=logger, fmt=fmt, level="DEBUG")

        elif verbose >= 1:
            logger.setLevel(logging.INFO)
            coloredlogs.install(logger=logger, fmt=fmt, level="INFO")
        else:
            streamhandler.setLevel(logging.CRITICAL)
            fileHandler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)

        logger.warning(f"Log file created at {log_file}")
        logger.debug(f"verbose {verbose}")

    return logger
