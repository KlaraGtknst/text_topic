import datetime
import logging
from constants import Paths
from utils.os_manipulation import exists_or_create


def get_date():
    return datetime.datetime.now().strftime('%x').replace('/', '_')


def init_debug_config(log_filename: str, on_server: bool = True, debug_level: int = logging.INFO):
    """
    This function initializes the logging configuration.
    :param log_filename: Name of the log file
    :param on_server: Boolean indicating whether the code is running on the server or locally
    :param debug_level: Level of debugging: DEBUG, INFO, WARNING, ERROR, CRITICAL
    :return: -
    """
    # a: write and append to log file
    # DEBUG: Detailed information, INFO: Confirmation that things are working as expected,
    # WARNING: An indication that something unexpected happened, ERROR: serious problem
    # CRITICAL: A serious error, program itself may be unable to continue running
    debugging_prefix = Paths.SERVER_LOGGING_PATH.value if on_server else Paths.LOCAL_LOGGING_PATH.value
    exists_or_create(debugging_prefix)

    # create logger
    logging.basicConfig(level=debug_level, filename=f'{debugging_prefix}{log_filename}{get_date()}.log',
                        filemode='a', format='%(asctime)s %(message)s')
