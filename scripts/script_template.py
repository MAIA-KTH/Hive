#!/usr/bin/env python

import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script Description
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} Script usage examples.
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-arg",
        "--argument",
        type=str,
        required=True,
        help="Example argument",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


if __name__ == "__main__":
    main()
