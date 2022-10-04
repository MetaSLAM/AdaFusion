from .aavislidarnet3 import *


def get_model(config):
    """Return the specific model.
    Args:
        config: An EasyDict config object from utils.load_config()
    Returns:
        model: The specific model.
    """
    return globals()[config.architecture](config.fusion_method, config.loss.Lp)
