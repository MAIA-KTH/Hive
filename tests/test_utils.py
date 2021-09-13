import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest

from Hive.utils.log_utils import (
    add_verbosity_options_to_argparser,
    get_logger,
    log_lvl_from_verbosity_args,
    DEBUG,
    DEBUG2,
    WARN,
)


def get_test_logger(message_level, log_format):
    parser = ArgumentParser()

    parser.add_argument(
        "--string",
        type=str,
    )
    add_verbosity_options_to_argparser(parser)
    arguments = vars(parser.parse_args())
    logger = get_logger(
        name="test_logger",
        level=log_lvl_from_verbosity_args(arguments),
        fmt=log_format,
    )
    logger.log(message_level, arguments["string"])


@pytest.mark.parametrize(
    "verbosity_level,message_level,log_format",
    [
        (None, DEBUG, "%(message)s"),
        ("-v", DEBUG, "%(asctime)s - %(message)s"),
        ("-vv", DEBUG2, "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        (None, WARN, "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        ("-q", WARN, "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    ],
)
def test_argparser_logger(verbosity_level, message_level, log_format):
    sys.argv = [Path(__file__).name, "--string", "Hello World"]
    if verbosity_level is not None:
        sys.argv.append(verbosity_level)
    get_test_logger(message_level, log_format)
