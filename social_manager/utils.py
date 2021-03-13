import logging
import os
import sys

from .config_reader import ConfigReader

_SOCIAL_MANAGER_CFG = ConfigReader("configs/social_manager.cfg")


def _gen_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_data_dir():
    _gen_dir(_SOCIAL_MANAGER_CFG["data_dir"])

    return _SOCIAL_MANAGER_CFG["data_dir"]


def get_log_dir():
    _gen_dir(_SOCIAL_MANAGER_CFG["log_dir"])

    return _SOCIAL_MANAGER_CFG["log_dir"]


def setup_logger(logger, stream_level=logging.INFO, file_level=logging.DEBUG):
    file_path = os.path.join(get_log_dir(), "social_manager.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)06d: %(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(stream_level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(min(stream_level, file_level))
