import os
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