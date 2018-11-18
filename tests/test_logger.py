import pytest
from core.utils.logger import loggerConfig
import logging

def test_logger_verbose1():
    logger = loggerConfig('test.log', 1, 'namelogger1')
    assert logger.level == logging.INFO

def test_logger_verbose0():
    logger = loggerConfig('test.log', 0, 'namelogger2')
    assert isinstance(logger.handlers[-1], logging.StreamHandler) and logger.handlers[-1].level == logging.CRITICAL

def test_logger_verbose2():
    logger = loggerConfig('test.log', 2, 'namelogger3')
    print(logger.handlers)
    assert logger.level == logging.DEBUG

