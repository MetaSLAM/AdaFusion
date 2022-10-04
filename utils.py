import logging
import math
import os
import shutil
import yaml
from easydict import EasyDict

import torch


def get_logger(log_file_name, log_level, logger_name):
    """return an easy-to-use logger to record infomation
    Args:
        log_file_name: The file to save log.
        log_level: Over which level will the messages be logged. Can be
            int or str, logging.DEBUG / "DEBUG", 'INFO', 'WARNING', 'ERROR',
            'FATAL', ...
        logger_name: The name of the logger. It can have parent-child relationship
            using period for seperation, e.g. parent.child. See docs.
    Returns:
        logger: The logger object
    """
    logger = logging.getLogger(logger_name)

    file_handler = logging.FileHandler(log_file_name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        # "[%(levelname)s %(asctime)s]-[%(filename)s line:%(lineno)3d] : %(message)s"
        "[%(levelname)s %(asctime)s] : %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(log_level)

    return logger


def load_config(config_file_name):
    """Read config from yaml file. You can access item using dict-like or
    attribute-like ways, i.e. config['xxx'] or config.xxx
    Args:
        config_file_name: the *.yaml config file name
    Returns:
        config: An EasyDict config object
    """
    with open(config_file_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)  # convert to dict


def count_parameters(model):
    """Count the number of variables that require gradient in the net model.
    This is because only variables requiring gradient will change during training.
    Args:
        model: net model, sub-class of nn.Module
    Returns:
        number: int, the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + "_best.pth")


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint {"state_dict","best_prec","last_epoch","optimizer"} from file.
    Args:
        path: The path to a checkpoint file, xxx.tar
        model: Network nn.Module
        optimizer: If not None, also load optimizer params and return `best_prec` and
            `last_epoch`. If None, only load the net param.
    Returns:
        best_prec: best test acc. If optimizer is not None.
        last_epoch: the last epoch number. If optimizer is not None.
    Raises:
        ValueError: `path` is not a valid checkpoint file path.
    """
    logger = logging.getLogger("CIFAR")

    if os.path.isfile(path):
        logger.info(f"=== loading checkpoint '{path}' ===")

        checkpoint = torch.load(path)  # a custom dict
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        if optimizer is not None:
            best_prec = checkpoint["best_prec"]
            last_epoch = checkpoint["last_epoch"] - 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=== done. Also loaded optimizer from "
                f"checkpoint '{path}' (epoch {last_epoch + 1}), best={best_prec:.4} ==="
            )
            return best_prec, last_epoch
    else:
        logger.error(f"file {path} is NOT a valid checkpoint path!!")
        raise ValueError


def get_current_lr(optimizer):
    """Get current learning rate from the `optimizer`.
    optimizer.param_groups is a list of param_group, while param_group is a dict
    of {'params':xxx, 'lr':xx, 'momentum':xx, ....} representing actions for these
    parameters.
    Args:
        optimizer: the torch.optim.XXX object
    Returns:
        lr: learning rate of the first layer of the optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def adjust_learning_rate(optimizer, epoch, config):
    """Adjust learning rate with different methods. Actually this can be done using the official
    function `torch.optim.lr_scheduler`, see https://pytorch.org/docs/stable/optim.html
    """
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == "STEP":
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == "COSINE":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (1.0 + math.cos(math.pi * ratio))
            / 2.0
        )
    elif config.lr_scheduler.type == "HTD":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (
                1.0
                - math.tanh(
                    config.lr_scheduler.lower_bound
                    + (
                        config.lr_scheduler.upper_bound
                        - config.lr_scheduler.lower_bound
                    )
                    * ratio
                )
            )
            / 2.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def freeze_partial_net(names: list, net: torch.nn.Module):
    """Freeze part of parameters of the network.
    This is useful when 1. fine tuning, 2. first train the backbone and the other part.
    Args:
        names: A list of param names that you want to freeze, e.g. ['v_attention','l_attention'].
            the matching pattern is 'str in param_name'.
        net: the pytorch network
    Returns: None
    """
    for param_name, param in net.named_parameters():
        for name in names:
            if name in param_name:
                param.requires_grad = False
                break
