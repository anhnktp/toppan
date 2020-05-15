import os
import logging
import warnings
import multiprocessing_logging
from dotenv import load_dotenv
from os.path import join, dirname, abspath

def setup_logger(name, is_log_file, log_file, formatter, level=logging.INFO, filemode='w'):
    """To setup as many loggers as you want"""
    if is_log_file == 'TRUE':
        handler = logging.FileHandler(log_file, filemode)
    else: handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Config ignore warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
logging.captureWarnings(True)

# Config dotenv
ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
dotenv_path = join(ROOT_DIR, '.env')
load_dotenv(dotenv_path)

# Config logging
multiprocessing_logging.install_mp_handler()

# bpo logger
bpo_logger = setup_logger('bpo_logger', os.getenv('LOG_FILE'), join(ROOT_DIR, 'logs', 'bpo.log'),
                          logging.Formatter('[%(levelname)s|%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
# engine logger
engine_logger = setup_logger('engine_logger', os.getenv('LOG_FILE'), join(ROOT_DIR, 'logs', 'engine.log'),
                             logging.Formatter('[%(levelname)s|%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
