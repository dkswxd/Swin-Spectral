# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger

current_split = -1

def get_root_logger(log_file=None, log_level=logging.INFO, split=-1):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    global current_split
    if current_split == -1 and split == -1:
        logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)
    elif split != -1:
        current_split = split
        logger = get_logger(name=f'mmseg_{current_split}', log_file=log_file, log_level=log_level)
    else:
        logger = get_logger(name=f'mmseg_{current_split}', log_file=log_file, log_level=log_level)

    return logger
