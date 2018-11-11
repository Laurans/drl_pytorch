import logging


from logging import RootLogger


def loggerConfig(log_file: str, verbose: int) -> RootLogger:
    logger = logging.getLogger()
    logger.propagate = 0
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(levelname)-8s] (%(module)s - %(funcName)s) %(message)s"
        )

        fileHandler = logging.FileHandler(log_file, "w")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)

    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.NOTSET)

    return logger
