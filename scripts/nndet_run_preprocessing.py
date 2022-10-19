#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Run nnDetection pre-processing, data analysis, and image data unpacking.
    The CL scripts called are  ``nndet_prep`` and ``nndet_unpack``, with the arguments extracted from the given configuration file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /PATH/TO/CONFIG_FILE.json
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiment variables (**Task_ID**, **results_folder**, ...) ",
        # noqa: E501
    )

    pars.add_argument(
        "--n-workers",
        type=str,
        default=None,
        help="Number of parallel processes used when pre-processing and unpacking the image data (Default: ``N_THREADS``)",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )
    config_file = args["config_file"]

    n_workers = "1"
    if args["n_workers"] is None:
        if "N_THREADS" in os.environ is not None:
            n_workers = str(os.environ["N_THREADS"])
    else:
        n_workers = str(args["n_workers"])

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [data["Task_ID"], "-np", n_workers, "-npp", n_workers]

        os.environ["det_data"] = data["base_folder"]
        os.environ["OMP_NUM_THREADS"] = "1"

        os.environ["nnUNet_def_n_proc"] = n_workers
        os.environ["det_models"] = data["results_folder"]
        arguments.extend(unknown_arguments)
        os.system("nndet_prep " + " ".join(arguments))
        os.system(
            "nndet_unpack {} {}".format(
                str(
                    Path(os.environ["det_data"]).joinpath(
                        "Task{}_{}".format(data["Task_ID"], data["Task_Name"]), "preprocessed", "D3V001_3d", "imagesTr"
                    )
                ),
                n_workers,
            )
        )


if __name__ == "__main__":
    main()
