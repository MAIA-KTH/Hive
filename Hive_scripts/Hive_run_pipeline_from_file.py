#!/usr/bin/env python

import datetime
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, INFO

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Run pipeline steps from a TXT file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --file PIPELINE_FILE.txt
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--file",
        type=str,
        required=True,
        help="TXT file including list of commands to run.",
    )

    pars.add_argument(
        "--steps",
        type=str,
        nargs="+",
        required=False,
        help="Optional pipeline steps to run. When omitted, run all the steps.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(  # noqa: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    with open(arguments["file"]) as f:
        commands = f.readlines()

    steps = range(len(commands))
    if arguments["steps"] is not None:
        steps = arguments["steps"]
        steps = [int(step) for step in steps]

    for it, command in enumerate(commands):
        if it in steps:
            logger.log(INFO, "Running Step {}: {}".format(it, command))
            subprocess.call(command[:-1].split(" "))


if __name__ == "__main__":
    main()
