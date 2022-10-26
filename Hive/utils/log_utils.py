import argparse
import logging
from argparse import ArgumentParser
from typing import Dict, Union

import coloredlogs

__VERBOSITY_GRANULARITY__ = 2

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# define custom log levels
DEBUG = logging.DEBUG
DEBUG2 = DEBUG - 2
DEBUG4 = DEBUG - 4
DEBUG6 = DEBUG - 6
WARN = logging.WARN
INFO = logging.INFO
# add levels to the 'DEBUG' group (i.e. same coloring)
for lvl_value in [DEBUG2, DEBUG4, DEBUG6]:
    logging.addLevelName(lvl_value, "DEBUG")


def setup_logging(level: Union[str, int] = "DEBUG", fmt: str = LOG_FMT):
    coloredlogs.install(level=level, fmt=fmt)


def get_logger(name: str = "", level: Union[str, int] = None, fmt: str = LOG_FMT) -> logging.Logger:
    """Get a logger and setup colored logging if level is provided."""

    if level is not None:
        setup_logging(level=level, fmt=fmt)

    logger = logging.getLogger(name)

    return logger


def add_verbosity_options_to_argparser(p: ArgumentParser):
    gp = p.add_mutually_exclusive_group(required=False)
    gp.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="count",
        help="Enable verbose output in terminal. Add multiple times to increase verbosity.",  # noqa: E501
    )
    gp.add_argument(
        "-q",
        "--silent",
        required=False,
        action="count",
        help="Suppress most log outputs in terminal.",
    )


def log_lvl_from_verbosity_args(args: Dict) -> int:
    if args["verbose"]:
        return logging.DEBUG - __VERBOSITY_GRANULARITY__ * (args["verbose"] - 1)  # noqa: 501
    elif args["silent"]:
        return logging.WARN
    else:
        return logging.INFO


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
